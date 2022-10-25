#!/bin/bash
mpirun -n 2 python -u -O -m mpi4py wall.py -i run_params.yaml -r restart_data/wall-000000010 --log --lazy
