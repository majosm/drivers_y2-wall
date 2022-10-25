#!/bin/bash
gmsh -setnumber size 0.0032 -setnumber blratio 4 -setnumber injectorfac 5 -o actii_2d.msh -nopopup -format msh2 ./actii_2d.geo -2
