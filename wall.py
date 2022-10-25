"""mirgecom driver for the Y2 wall."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import logging
import sys
import yaml
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
import math
from dataclasses import dataclass, fields
from functools import partial
from mirgecom.discretization import create_discretization_collection

from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import BoundaryDomainTag, VTAG_ALL
from grudge.op import nodal_max, nodal_min
from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    logmgr_add_device_memory_usage
)

from mirgecom.simutil import (
    check_step,
    distribute_mesh,
    write_visfile,
    check_range_local,
    force_evaluation
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import (rk4_step, lsrk54_step, lsrk144_step,
                                  euler_step)
from grudge.shortcuts import compiled_lsrk45_step

from mirgecom.steppers import advance_state
from mirgecom.diffusion import (
    grad_operator,
    diffusion_operator,
    DirichletDiffusionBoundary
)


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class _EnergyDiffCommTag:
    pass


class _OxDiffCommTag:
    pass


def mask_from_elements(vol_discr, actx, elements):
    mesh = vol_discr.mesh
    zeros = vol_discr.zeros(actx)

    group_arrays = []

    for igrp in range(len(mesh.groups)):
        start_elem_nr = mesh.base_element_nrs[igrp]
        end_elem_nr = start_elem_nr + mesh.groups[igrp].nelements
        grp_elems = elements[
            (elements >= start_elem_nr)
            & (elements < end_elem_nr)] - start_elem_nr
        grp_ary_np = actx.to_numpy(zeros[igrp])
        grp_ary_np[grp_elems] = 1
        group_arrays.append(actx.from_numpy(grp_ary_np))

    return DOFArray(actx, tuple(group_arrays))


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class WallVars:
    mass: DOFArray
    energy: DOFArray
    ox_mass: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return get_container_context_recursively(self.mass)

    def __reduce__(self):
        """Return a tuple reproduction of self for pickling."""
        return (WallVars, tuple(getattr(self, f.name)
                                    for f in fields(WallVars)))


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
    thermal_conductivity: DOFArray
    temperature: DOFArray


class WallModel:
    """Model for calculating wall quantities."""
    def __init__(
            self,
            heat_capacity,
            thermal_conductivity_func,
            *,
            effective_surface_area_func=None,
            mass_loss_func=None,
            oxygen_diffusivity=0.):
        self._heat_capacity = heat_capacity
        self._thermal_conductivity_func = thermal_conductivity_func
        self._effective_surface_area_func = effective_surface_area_func
        self._mass_loss_func = mass_loss_func
        self._oxygen_diffusivity = oxygen_diffusivity

    @property
    def heat_capacity(self):
        return self._heat_capacity

    def thermal_conductivity(self, mass, temperature):
        return self._thermal_conductivity_func(mass, temperature)

    def thermal_diffusivity(self, mass, temperature, thermal_conductivity=None):
        if thermal_conductivity is None:
            thermal_conductivity = self.thermal_conductivity(mass, temperature)
        return thermal_conductivity/(mass * self.heat_capacity)

    def mass_loss_rate(self, mass, ox_mass, temperature):
        dm = mass*0.
        if self._effective_surface_area_func is not None:
            eff_surf_area = self._effective_surface_area_func(mass)
            if self._mass_loss_func is not None:
                dm = self._mass_loss_func(mass, ox_mass, temperature, eff_surf_area)
        return dm

    @property
    def oxygen_diffusivity(self):
        return self._oxygen_diffusivity

    def temperature(self, wv):
        return wv.energy/(wv.mass * self.heat_capacity)

    def dependent_vars(self, wv):
        temperature = self.temperature(wv)
        kappa = self.thermal_conductivity(wv.mass, temperature)
        return WallDependentVars(
            thermal_conductivity=kappa,
            temperature=temperature)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context,
         restart_filename=None, target_filename=None,
         use_profiling=False, use_logmgr=True, user_input_file=None,
         use_overintegration=False, actx_class=None, casename=None,
         lazy=False):

    if actx_class is None:
        raise RuntimeError("Array context class missing.")

    # control log messages
    logger = logging.getLogger(__name__)
    logger.propagate = False

    if (logger.hasHandlers()):
        logger.handlers.clear()

    # send info level messages to stdout
    h1 = logging.StreamHandler(sys.stdout)
    f1 = SingleLevelFilter(logging.INFO, False)
    h1.addFilter(f1)
    logger.addHandler(h1)

    # send everything else to stderr
    h2 = logging.StreamHandler(sys.stderr)
    f2 = SingleLevelFilter(logging.INFO, True)
    h2.addFilter(f2)
    logger.addHandler(h2)

    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    if casename is None:
        casename = "mirgecom"

    # logging and profiling
    log_path = "log_data/"
    #log_path = ""
    logname = log_path + casename + ".sqlite"

    if rank == 0:
        import os
        log_dir = os.path.dirname(logname)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    comm.Barrier()

    logmgr = initialize_logmgr(use_logmgr,
        filename=logname, mode="wo", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    # main array context for the simulation
    from mirgecom.simutil import get_reasonable_memory_pool
    alloc = get_reasonable_memory_pool(cl_ctx, queue)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000, allocator=alloc)
    else:
        actx = actx_class(comm, queue, allocator=alloc, force_device_scalars=True)

    # default i/o junk frequencies
    nviz = 500
    nhealth = 1
    nrestart = 5000
    nstatus = 1
    # verbosity for what gets written to viz dumps, increase for more stuff
    viz_level = 1
    # control the time interval for writing viz dumps
    viz_interval_type = 0

    # default timestepping control
    integrator = "rk4"
    current_dt = 1e-8
    t_final = 1e-7
    t_viz_interval = 1.e-8
    current_t = 0
    t_start = 0.
    current_step = 0
    current_cfl = 1.0
    constant_cfl = False
    last_viz_interval = 0
    force_eval = True

    # discretization and model control
    order = 1
    dim = 2
    mesh_filename = "data/actii_2d.msh"

    # material properties
    spec_diff = 1.e-4

    # Averaging from https://www.azom.com/article.aspx?ArticleID=1630
    # for graphite
    insert_rho = 1625
    insert_cp = 770
    insert_kappa = 247.5  # This seems high

    # Fiberform
    # insert_rho = 183.6
    # insert_cp = 710
    insert_ox_diff = spec_diff

    # Averaging from http://www.matweb.com/search/datasheet.aspx?bassnum=MS0001
    # for steel
    surround_rho = 7.9e3
    surround_cp = 470
    surround_kappa = 48

    # rhs control
    use_ox = True
    use_mass = True

    # wall stuff
    penalty_amount = 25
    material = 0

    # Surface boundary values
    surface_temperature = 600.0
    surface_ox_mass = 0.25

    if user_input_file:
        input_data = None
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        input_data = comm.bcast(input_data, root=0)
        try:
            nviz = int(input_data["nviz"])
        except KeyError:
            pass
        try:
            t_viz_interval = float(input_data["t_viz_interval"])
        except KeyError:
            pass
        try:
            viz_interval_type = int(input_data["viz_interval_type"])
        except KeyError:
            pass
        try:
            viz_level = int(input_data["viz_level"])
        except KeyError:
            pass
        try:
            nrestart = int(input_data["nrestart"])
        except KeyError:
            pass
        try:
            nhealth = int(input_data["nhealth"])
        except KeyError:
            pass
        try:
            nstatus = int(input_data["nstatus"])
        except KeyError:
            pass
        try:
            constant_cfl = int(input_data["constant_cfl"])
        except KeyError:
            pass
        try:
            current_dt = float(input_data["current_dt"])
        except KeyError:
            pass
        try:
            current_cfl = float(input_data["current_cfl"])
        except KeyError:
            pass
        try:
            t_final = float(input_data["t_final"])
        except KeyError:
            pass
        try:
            spec_diff = float(input_data["spec_diff"])
            insert_ox_diff = spec_diff
        except KeyError:
            pass
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            dim = int(input_data["dimen"])
        except KeyError:
            pass
        try:
            integrator = input_data["integrator"]
        except KeyError:
            pass
        try:
            use_ox = bool(input_data["use_ox"])
        except KeyError:
            pass
        try:
            use_mass = bool(input_data["use_mass"])
        except KeyError:
            pass
        try:
            mesh_filename = input_data["mesh_filename"]
        except KeyError:
            pass
        try:
            penalty_amount = float(input_data["penalty_amount"])
        except KeyError:
            pass
        try:
            material = int(input_data["material"])
        except KeyError:
            pass
        try:
            insert_rho = float(input_data["insert_rho"])
        except KeyError:
            pass
        try:
            insert_cp = float(input_data["insert_cp"])
        except KeyError:
            pass
        try:
            insert_kappa = float(input_data["insert_kappa"])
        except KeyError:
            pass
        try:
            surround_rho = float(input_data["surround_rho"])
        except KeyError:
            pass
        try:
            surround_cp = float(input_data["surround_cp"])
        except KeyError:
            pass
        try:
            surround_kappa = float(input_data["surround_kappa"])
        except KeyError:
            pass
        try:
            surface_temperature = float(input_data["surface_temperature"])
        except KeyError:
            pass
        try:
            surface_ox_mass = float(input_data["surface_ox_mass"])
        except KeyError:
            pass

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144", "compiled_lsrk54"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    if integrator == "compiled_lsrk54":
        print("Setting force_eval = False for pre-compiled time integration")
        force_eval = False

    if viz_interval_type > 2:
        error_message = "Invalid value for viz_interval_type [0-2]"
        raise RuntimeError(error_message)

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        if constant_cfl == 1:
            print(f"\tConstant cfl mode, current_cfl = {current_cfl}")
        else:
            print(f"\tConstant dt mode, current_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tdimen = {dim}")
        print(f"\tTime integration {integrator}")
        print("#### Simulation control data: ####\n")

    if rank == 0:
        print("\n#### Visualization setup: ####")
        if viz_level >= 0:
            print("\tBasic visualization output enabled.")
            print("\t(wv, dv, cfl)")
        if viz_level >= 1:
            print("\tExtra visualization output enabled for derived quantities.")
        if viz_level >= 2:
            print("\tNon-dimensional parameter visualization output enabled.")
            print("\t(Re, Pr, etc.)")
        if viz_level >= 3:
            print("\tDebug visualization output enabled.")
            print("\t(rhs, grad_t, etc.)")
        if viz_interval_type == 0:
            print(f"\tWriting viz data every {nviz} steps.")
        if viz_interval_type == 1:
            print(f"\tWriting viz data roughly every {t_viz_interval} seconds.")
        if viz_interval_type == 2:
            print(f"\tWriting viz data exactly every {t_viz_interval} seconds.")
        print("#### Visualization setup: ####")

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step
    if integrator == "compiled_lsrk54":
        timestepper = _compiled_stepper_wrapper

    # }}}
    # working gas: O2/N2 #
    #   O2 mass fraction 0.273
    #   gamma = 1.4
    #   cp = 37.135 J/mol-K,
    #   rho= 1.977 kg/m^3 @298K
    mw_o = 15.999
    mw_o2 = mw_o*2
    mw_co = 28.010
    univ_gas_const = 8314.59

    init_temperature = 300.0

    if rank == 0:
        print("\n#### Simulation material properties: ####")
        if material == 0:
            print("\tNon-reactive wall model")
        elif material == 1:
            print("\tReactive wall model for non-porous media")
        elif material == 2:
            print("\tReactive wall model for porous media")
        else:
            error_message = "Unknown material {}".format(material)
            raise RuntimeError(error_message)

        if use_ox:
            print("\tWall oxidizer transport enabled")
        else:
            print("\tWall oxidizer transport disabled")

        if use_mass:
            print("\t Wall mass loss enabled")
        else:
            print("\t Wall mass loss disabled")

        print(f"\tWall density = {insert_rho}")
        print(f"\tWall cp = {insert_cp}")
        print(f"\tWall O2 diff = {insert_ox_diff}")
        print(f"\tWall surround density = {surround_rho}")
        print(f"\tWall surround cp = {surround_cp}")
        print(f"\tWall surround kappa = {surround_kappa}")
        print(f"\tWall penalty = {penalty_amount}")
        print("#### Simulation material properties: ####")

    viz_path = "viz_data/"
    vizname = viz_path + casename
    restart_path = "restart_data/"
    restart_pattern = (
        restart_path + "{cname}-{step:09d}-{rank:04d}.pkl"
    )

    if restart_filename:  # read the grid from restart data
        restart_filename = f"{restart_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_filename)
        current_step = restart_data["step"]
        current_t = restart_data["t"]
        last_viz_interval = restart_data["last_viz_interval"]
        t_start = current_t
        local_mesh_data = restart_data["local_mesh_data"]
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert restart_data["nparts"] == nparts
    else:  # generate the grid from scratch
        if rank == 0:
            print(f"Reading mesh from {mesh_filename}")

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_filename, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            from mirgecom.simutil import extract_volumes
            wall_mesh, tag_to_wall_elements = extract_volumes(
                mesh, tag_to_elements,
                selected_tags=["wall_insert", "wall_surround"],
                boundary_tag="surface")
            volume_to_tags = {
                VTAG_ALL: ["wall_insert", "wall_surround"]}
            return wall_mesh, tag_to_wall_elements, volume_to_tags

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data)
        local_mesh_data = volume_to_local_mesh_data[VTAG_ALL]

    local_mesh, tag_to_elements = local_mesh_data
    local_nelements = local_mesh.nelements

    if rank == 0:
        logger.info("Making discretization")

    dcoll = create_discretization_collection(
        actx,
        volume_meshes=local_mesh,
        order=order)

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    vol_discr = dcoll.discr_from_dd("vol")
    insert_mask = mask_from_elements(
        vol_discr, actx, tag_to_elements["wall_insert"])
    surround_mask = mask_from_elements(
        vol_discr, actx, tag_to_elements["wall_surround"])

    from grudge.dt_utils import characteristic_lengthscales
    char_length = characteristic_lengthscales(actx, dcoll)

    if rank == 0:
        logger.info("Before restart/init")

    #########################
    # Convenience Functions #
    #########################

    def _create_dependent_vars(wv):
        return wall_model.dependent_vars(wv)

    create_dependent_vars_compiled = actx.compile(
        _create_dependent_vars)

    def _get_wv(wv):
        return wv

    get_wv = actx.compile(_get_wv)

    if restart_filename:
        if rank == 0:
            logger.info("Restarting soln.")
        restart_wv = restart_data["wv"]
        if restart_order != order:
            restart_dcoll = create_discretization_collection(
                actx,
                local_mesh,
                order=restart_order)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd("vol"),
                restart_dcoll.discr_from_dd("vol")
            )
            restart_wv = connection(restart_data["wv"])

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        if rank == 0:
            logger.info("Initializing soln.")
        mass = (
            insert_rho * insert_mask
            + surround_rho * surround_mask)
        cp = (
            insert_cp * insert_mask
            + surround_cp * surround_mask)
        restart_wv = WallVars(
            mass=mass,
            energy=mass * cp * init_temperature,
            ox_mass=0*mass)

    current_wv = force_evaluation(actx, restart_wv)
    #current_wv = get_wv(restart_wv)

    stepper_state = current_wv

    ##################################
    # Set up the boundary conditions #
    ##################################

    surface = DirichletDiffusionBoundary(surface_temperature)
    farfield = DirichletDiffusionBoundary(init_temperature)

    boundaries = {
        BoundaryDomainTag("surface"): surface,
        BoundaryDomainTag("wall_farfield"): farfield
    }

    if use_ox:
        ox_surface = DirichletDiffusionBoundary(surface_ox_mass)
        ox_farfield = DirichletDiffusionBoundary(0)
        ox_boundaries = {
            BoundaryDomainTag("surface"): ox_surface,
            BoundaryDomainTag("wall_farfield"): ox_farfield
        }

    def _grad_t_operator(t, kappa, temperature):
        return grad_operator(
            dcoll, kappa, boundaries, temperature,
            quadrature_tag=quadrature_tag)

    grad_t_operator = actx.compile(_grad_t_operator)

    def experimental_kappa(temperature):
        return (
            1.766e-10 * temperature**3
            - 4.828e-7 * temperature**2
            + 6.252e-4 * temperature
            + 6.707e-3)

    def puma_kappa(mass_loss_frac):
        return (
            0.0988 * mass_loss_frac**2
            - 0.2751 * mass_loss_frac
            + 0.201)

    def puma_effective_surface_area(mass_loss_frac):
        # Original fit function: -1.1012e5*x**2 - 0.0646e5*x + 1.1794e5
        # Rescale by x==0 value and rearrange
        return 1.1794e5 * (
            1
            - 0.0547736137 * mass_loss_frac
            - 0.9336950992 * mass_loss_frac**2)

    def _get_kappa_fiber(mass, temperature):
        mass_loss_frac = (
            (insert_rho - mass)/insert_rho
            * insert_mask)
        scaled_insert_kappa = (
            experimental_kappa(temperature)
            * puma_kappa(mass_loss_frac)
            / puma_kappa(0))
        return (
            scaled_insert_kappa * insert_mask
            + surround_kappa * surround_mask)

    def _get_kappa_inert(mass, temperature):
        return (
            insert_kappa * insert_mask
            + surround_kappa * surround_mask)

    def _get_effective_surface_area_fiber(mass):
        mass_loss_frac = (
            (insert_rho - mass)/insert_rho
            * insert_mask)
        return (
            puma_effective_surface_area(mass_loss_frac) * insert_mask)

    def _mass_loss_rate_fiber(mass, ox_mass, temperature, eff_surf_area):
        actx = mass.array_context
        alpha = (
            (0.00143+0.01*actx.np.exp(-1450.0/temperature))
            / (1.0+0.0002*actx.np.exp(13000.0/temperature)))
        k = alpha*actx.np.sqrt(
            (univ_gas_const*temperature)/(2.0*np.pi*mw_o2))
        return (mw_co/mw_o2 + mw_o/mw_o2 - 1)*ox_mass*k*eff_surf_area

    # inert
    if material == 0:
        wall_model = WallModel(
            heat_capacity=(
                insert_cp * insert_mask
                + surround_cp * surround_mask),
            thermal_conductivity_func=_get_kappa_inert)
    # non-porous
    elif material == 1:
        wall_model = WallModel(
            heat_capacity=(
                insert_cp * insert_mask
                + surround_cp * surround_mask),
            thermal_conductivity_func=_get_kappa_fiber,
            effective_surface_area_func=_get_effective_surface_area_fiber,
            mass_loss_func=_mass_loss_rate_fiber,
            oxygen_diffusivity=insert_ox_diff * insert_mask)
    # porous
    elif material == 2:
        wall_model = WallModel(
            heat_capacity=(
                insert_cp * insert_mask
                + surround_cp * surround_mask),
            thermal_conductivity_func=_get_kappa_fiber,
            effective_surface_area_func=_get_effective_surface_area_fiber,
            mass_loss_func=_mass_loss_rate_fiber,
            oxygen_diffusivity=insert_ox_diff * insert_mask)

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "step walltime: {value:6g} s, ")
            #("t_log.max", "log walltime: {value:6g} s")
        ])

        try:
            logmgr.add_watches(["memory_usage_python.max"])
            #logmgr.add_watches(["memory_usage_python.max", "memory: {value}"])
        except KeyError:
            pass

        try:
            logmgr.add_watches(["memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(dcoll)

    #    initname = initializer.__class__.__name__
    init_message = make_init_message(dim=dim, order=order, nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=casename,
                                     eosname="None", casename=casename)
    if rank == 0:
        logger.info(init_message)

    # some utility functions
    def vol_min_loc(x):
        from grudge.op import nodal_min_loc
        return actx.to_numpy(nodal_min_loc(dcoll, "vol", x,
                                           initial=np.inf))[()]

    def vol_max_loc(x):
        from grudge.op import nodal_max_loc
        return actx.to_numpy(nodal_max_loc(dcoll, "vol", x,
                                           initial=-np.inf))[()]

    def vol_min(x):
        return actx.to_numpy(nodal_min(dcoll, "vol", x,
                                       initial=np.inf))[()]

    def vol_max(x):
        return actx.to_numpy(nodal_max(dcoll, "vol", x,
                                       initial=-np.inf))[()]

    def global_range_check(array, min_val, max_val):
        return global_reduce(
            check_range_local(
                dcoll, "vol", array, min_val, max_val), op="lor")

    def my_write_status(temperature, dt, cfl):
        status_msg = (f"-------- dt = {dt:1.3e},"
                      f" cfl = {cfl:1.8f}")

        twmin = vol_min(temperature)
        twmax = vol_max(temperature)

        dv_status_msg = (
            f"\n-------- T_wall (min, max) (K)  = ({twmin:7g}, {twmax:7g})")

        status_msg += dv_status_msg
        status_msg += "\n"

        if rank == 0:
            logger.info(status_msg)

    def my_write_viz(step, t, wv, kappa, temperature, ts_field, dump_number):

        if rank == 0:
            print(f"******** Writing Visualization File {dump_number}"
                  f" at step {step},"
                  f" sim time {t:1.6e} s ********")

        # basic viz quantities, things here are difficult (or impossible) to compute
        # in post-processing
        viz_fields = [
            ("wv", wv),
            ("kappa", kappa),
            ("temperature", temperature),
            ("dt" if constant_cfl else "cfl", ts_field)
        ]

        # additional viz quantities, add in some non-dimensional numbers
        if viz_level > 1:
            cell_alpha = wall_model.thermal_diffusivity(
                wv.mass, temperature, kappa)

            viz_ext = [
                       ("alpha", cell_alpha)]
            viz_fields.extend(viz_ext)

        # debbuging viz quantities, things here are used for diagnosing run issues
        if viz_level > 2:
            grad_temperature = grad_t_operator(t, kappa, temperature)
            viz_ext = [("grad_temperature", grad_temperature)]
            viz_fields.extend(viz_ext)

        write_visfile(
            dcoll, viz_fields, visualizer,
            vizname=vizname, step=dump_number, t=t,
            overwrite=True, comm=comm)

        if rank == 0:
            print("******** Done Writing Visualization File ********\n")

    def my_write_restart(step, t, state):
        if rank == 0:
            print(f"******** Writing Restart File at step {step}, "
                  f"sim time {t:1.6e} s ********")

        wv = state
        restart_fname = restart_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "local_mesh_data": local_mesh_data,
                "wv": wv,
                "t": t,
                "step": step,
                "order": order,
                "last_viz_interval": last_viz_interval,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, restart_data, restart_fname, comm)

        if rank == 0:
            print("******** Done Writing Restart File ********\n")

    def my_health_check():
        # FIXME: Add health check
        return False

    def _my_get_timestep(
            dcoll, wv, kappa, temperature, t, dt, cfl, t_final,
            constant_cfl=False):
        """Return the maximum stable timestep for a typical heat transfer simulation.

        This routine returns *dt*, the users defined constant timestep, or *max_dt*,
        the maximum domain-wide stability-limited timestep for a fluid simulation.

        .. important::
            This routine calls the collective: :func:`~grudge.op.nodal_min` on the
            inside which makes it domain-wide regardless of parallel domain
            decomposition. Thus this routine must be called *collectively*
            (i.e. by all ranks).

        Two modes are supported:
            - Constant DT mode: returns the minimum of (t_final-t, dt)
            - Constant CFL mode: returns (cfl * max_dt)

        Parameters
        ----------
        dcoll
            Grudge discretization or discretization collection?
        t: float
            Current time
        t_final: float
            Final time
        dt: float
            The current timestep
        cfl: float
            The current CFL number
        constant_cfl: bool
            True if running constant CFL mode

        Returns
        -------
        float
            The dt (contant cfl) or cfl (constant dt) at every point in the mesh
        float
            The minimum stable cfl based on conductive heat transfer
        float
            The maximum stable DT based on conductive heat transfer
        """
        def get_timestep(wv, kappa, temperature):
            return (
                char_length*char_length
                / actx.np.maximum(
                    wall_model.thermal_diffusivity(
                        wv.mass, temperature, kappa),
                    wall_model.oxygen_diffusivity))

        actx = kappa.array_context
        mydt = dt
        if constant_cfl:
            from grudge.op import nodal_min
            ts_field = cfl*get_timestep(
                wv=wv, kappa=kappa, temperature=temperature)
            mydt = actx.to_numpy(
                nodal_min(
                    dcoll, "vol", ts_field, initial=np.inf))[()]
        else:
            from grudge.op import nodal_max
            ts_field = mydt/get_timestep(
                wv=wv, kappa=kappa, temperature=temperature)
            cfl = actx.to_numpy(
                nodal_max(
                    dcoll, "vol", ts_field, initial=0.))[()]

        return ts_field, cfl, mydt

    #my_get_timestep = actx.compile(_my_get_timestep)
    my_get_timestep = _my_get_timestep

    def _check_time(time, dt, interval, interval_type):
        toler = 1.e-6
        status = False

        dumps_so_far = math.floor((time-t_start)/interval)

        # dump if we just passed a dump interval
        if interval_type == 2:
            time_till_next = (dumps_so_far + 1)*interval - time
            steps_till_next = math.floor(time_till_next/dt)

            # reduce the timestep going into a dump to avoid a big variation in dt
            if steps_till_next < 5:
                dt_new = dt
                extra_time = time_till_next - steps_till_next*dt
                #if actx.np.abs(extra_time/dt) > toler:
                if abs(extra_time/dt) > toler:
                    dt_new = time_till_next/(steps_till_next + 1)

                if steps_till_next < 1:
                    dt_new = time_till_next

                dt = dt_new

            time_from_last = time - t_start - (dumps_so_far)*interval
            if abs(time_from_last/dt) < toler:
                status = True
        else:
            time_from_last = time - t_start - (dumps_so_far)*interval
            if time_from_last < dt:
                status = True

        return status, dt, dumps_so_far + last_viz_interval

    #check_time = _check_time

    def my_pre_step(step, t, dt, state):

        wv = state
        wdv = create_dependent_vars_compiled(wv)

        try:

            if logmgr:
                logmgr.tick_before()

            # disable non-constant dt timestepping for now
            # re-enable when we're ready

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)
            next_dump_number = step

            if any([do_viz, do_restart, do_health, do_status]):
                # compute the limited cv so we can viz what the rhs will actually see

                #print(wv)
                #wv = force_evaluation(actx, wv)
                #print(wv)
                # pass through, removes a bunch of tagging to avoid recomplie
                wv = get_wv(wv)
                #print(wv)

                if not force_eval:
                    wv = force_evaluation(actx, wv)

                ts_field, cfl, dt = my_get_timestep(
                    dcoll=dcoll, wv=wv, kappa=wdv.thermal_conductivity,
                    temperature=wdv.temperature, t=t, dt=dt,
                    cfl=current_cfl, t_final=t_final, constant_cfl=constant_cfl)

            if do_health:
                health_errors = global_reduce(my_health_check(), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.warning("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_status:
                my_write_status(dt=dt, cfl=cfl, temperature=wdv.temperature)

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                my_write_viz(
                    step=step, t=t, wv=wv, kappa=wdv.thermal_conductivity,
                    temperature=wdv.temperature, ts_field=ts_field,
                    dump_number=next_dump_number)

        except MyRuntimeError:
            if rank == 0:
                logger.error("Errors detected; attempting graceful exit.")

            if viz_interval_type == 0:
                dump_number = step
            else:
                dump_number = (math.floor((t-t_start)/t_viz_interval) +
                    last_viz_interval)

            my_write_viz(
                step=step, t=t, wv=wv, kappa=wdv.thermal_conductivity,
                temperature=wdv.temperature, ts_field=ts_field,
                dump_number=dump_number)
            my_write_restart(step=step, t=t, state=state)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        wv = state
        wdv = wall_model.dependent_vars(wv)

        energy_rhs = diffusion_operator(
            dcoll, wdv.thermal_conductivity, boundaries, wdv.temperature,
            penalty_amount=penalty_amount, quadrature_tag=quadrature_tag,
            comm_tag=_EnergyDiffCommTag)

        # mass loss
        mass_rhs = 0.*wv.mass
        if use_mass:
            mass_rhs = -wall_model.mass_loss_rate(
                mass=wv.mass, ox_mass=wv.ox_mass,
                temperature=wdv.temperature)

        # wall oxygen diffusion
        ox_mass_rhs = 0.*wv.ox_mass
        if use_ox:
            ox_mass_rhs = diffusion_operator(
                dcoll, wall_model.oxygen_diffusivity, ox_boundaries, wv.ox_mass,
                penalty_amount=penalty_amount, quadrature_tag=quadrature_tag,
                comm_tag=_OxDiffCommTag)

        return WallVars(
            mass=mass_rhs,
            energy=energy_rhs,
            ox_mass=ox_mass_rhs)

    current_step, current_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=current_dt,
                      t=current_t, t_final=t_final,
                      force_eval=force_eval,
                      state=stepper_state)
    current_wv = stepper_state
    current_wdv = create_dependent_vars_compiled(current_wv)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    ts_field, cfl, dt = my_get_timestep(dcoll=dcoll,
        wv=current_wv, kappa=current_wdv.thermal_conductivity,
        temperature=current_wdv.temperature, t=current_t, dt=current_dt,
        cfl=current_cfl, t_final=t_final, constant_cfl=constant_cfl)
    my_write_status(dt=dt, cfl=cfl, temperature=current_wdv.temperature)

    if viz_interval_type == 0:
        dump_number = current_step
    else:
        dump_number = (math.floor((current_t-t_start)/t_viz_interval) +
            last_viz_interval)

    my_write_viz(
        step=current_step, t=current_t, wv=current_wv,
        kappa=current_wdv.thermal_conductivity,
        temperature=current_wdv.temperature,
        ts_field=ts_field, dump_number=dump_number)
    my_write_restart(step=current_step, t=current_t, state=stepper_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    #root_logger = logging.getLogger()

    #logging.debug("A DEBUG message")
    #logging.info("An INFO message")
    #logging.warning("A WARNING message")
    #logging.error("An ERROR message")
    #logging.critical("A CRITICAL message")

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-t", "--target_file", type=ascii, dest="target_file",
                        nargs="?", action="store", help="simulation target file")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=False,
                        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")

    args = parser.parse_args()

    # for writing output
    casename = "wall"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")
    lazy = args.lazy
    if args.profile:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_filename}")

    target_filename = None
    if args.target_file:
        target_filename = (args.target_file).replace("'", "")
        print(f"Target file specified: {target_filename}")

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Using user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(restart_filename=restart_filename, target_filename=target_filename,
         user_input_file=input_file,
         use_profiling=args.profile, use_logmgr=args.log,
         use_overintegration=args.overintegration, lazy=lazy,
         actx_class=actx_class, casename=casename)

# vim: foldmethod=marker
