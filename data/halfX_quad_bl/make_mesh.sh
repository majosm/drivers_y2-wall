#!/bin/bash
NCPUS=$(getconf _NPROCESSORS_ONLN)
gmsh -setnumber scale 1 -setnumber scale_bl 2 -setnumber blratio 4 -setnumber cavityfac 8 -setnumber isofac 2 -setnumber shearfac 6 -setnumber blratiocavity 2 -setnumber blratiosample 2 -setnumber blratioinjector 2 -setnumber injectorfac 5 -o actii_2d.msh -nopopup -format msh2 ./actii_2d.geo -2 -nt $NCPUS
