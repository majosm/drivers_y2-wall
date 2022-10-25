# CEESD_y2-wall

Prediction wall case using [MIRGE-Com](https://github.com/illinois-ceesd/mirgecom)

Driver and associated tools for simulating the wall portion of the ACT-II facility in direct-connect configuration.
The simulation is 2D or 3D, and supports multiple insert materials.

The driver can be found in wall.py.

## Directory structure

The main driver is wall.py with the problem setup in [baseline](baseline) being the generally accepted way to run the simulation.

Simulation data (i.e. meshes) are located in [data](data)

Numerical experiments and/or driver variations can be located in [experiments](experiments), these are variations that may or may not derive from the current driver in baseline, although they generally have common ancestery.

The driver/data used to create the timing data is located in [timing_runs](timing_runs), and is a smaller version of the full baseline run.

Nightly CI is performed using the directory [smoke_test](smoke_test)

## Installation

```
./buildMirge.sh
```

Will checkout and build a local copy of emirge, complete with all the needed subpackages. A new conda environemnt will be created named mirgeDriver.Y2wall. 

### Additional options

```
./buildMirge.sh -h
```

There are several optional build parameters that are detailed further

### Archiving MIRGE-Com version information

```
./updateVersionInfo.sh
```

Save the current build state of MIRGE-Com and associated packages into [platforms](platforms). Setting the environment variable MIRGE_PLATFORM will all the user to retrieve this version information when building MIRGE-Com using the buildMirge.sh script.

## Building the mesh

### Install gmsh
In the emirge directory activate the ceesd environment and install gmsh
```
conda activate mirgeDriver.Y2wall
conda install gmsh
```

### Run gmsh
In the directory containing the case file generate the mesh
```
gmsh -o prediction.msh -nopopup -format msh2 ./prediction.geo -2
```

Additional options may be available for fine-tuning mesh generation. Most mesh directories contain a script make_mesh.sh demonstrating how to build a particular mesh version.

## Running a case

Activate the correct conda environment
```
conda activate mirgeDriver.Y2wall
```

Most subdirectories contain a run.sh script that outlines how to run the problem.

The case can the be run similar to other MIRGE-Com applications.
For examples see the MIRGE-Com [documentation](https://mirgecom.readthedocs.io/en/latest/running/systems.html)
