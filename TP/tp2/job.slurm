#!/bin/bash
#
#SBATCH --job-name=test_mpi
#SBATCH --output=res_mpi.txt
#
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=1G

module load OpenMPI

mpicxx -O3 distributed_student.cpp -o distributed && mpirun -np 4 ./distributed 
