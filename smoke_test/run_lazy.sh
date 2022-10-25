#!/bin/bash
#mpirun -n 1 python -u -O -m mpi4py wall.py -i run_params.yaml --log
mpirun -n 2 python -u -O -m mpi4py wall.py -i run_params.yaml --log --lazy
