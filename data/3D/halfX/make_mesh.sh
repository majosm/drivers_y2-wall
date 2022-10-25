#!/bin/bash

NCPUS=$(getconf _NPROCESSORS_ONLN)
gmsh -setnumber size 1.6 -setnumber blratio 4 -setnumber cavityfac 4 -setnumber isofac 2 -setnumber injectorfac 6 -setnumber blratiocavity 2 -setnumber blratioinjector 2 -setnumber blratiosample 4 -setnumber blratiosurround 2 -setnumber shearfac 4 -setnumber -o actii_3d.msh -nopopup -format msh2 ./actii_3d_from_brep.geo -3 -nt $NCPUS
