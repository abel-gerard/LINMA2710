#!/bin/sh
#
#SBATCH --job-name=matrix_benchmark
#SBATCH --partition=gpu
#
#SBATCH --time=4:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:1

module purge
module load OpenMPI
module load CUDA

export DIM_COUNT=19 && ./command.sh
