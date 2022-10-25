#!/bin/bash
#gmsh -setnumber size 0.0064 -setnumber blratio 4 -setnumber injectorfac 5 -o actii_3d.msh -nopopup -format msh2 ./actii_3d_from_brep.geo -3 -nt $NCPUS
#gmsh -setnumber size 0.0064 -setnumber blratio 4 -setnumber injectorfac 5 -o actii_3d.msh -nopopup -format msh2 ./actii_3d_from_brep.geo -3
#gmsh -setnumber size 6.4 -setnumber blratio 4 -setnumber injectorfac 5 -o actii_3d.msh -nopopup -format msh2 ./actii_3d_from_brep.geo -3 -nt 10
gmsh -setnumber size 6.4 -setnumber blratio 1 -setnumber injectorfac 5 -setnumber blratiocavity 2 -setnumber blratioinjector 2 -setnumber blratiosample 4 -setnumber blratiosurround 2 -o actii_3d.msh -nopopup -format msh2 ./actii_3d_from_brep.geo -3 -nt 5
