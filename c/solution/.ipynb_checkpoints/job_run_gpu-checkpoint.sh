#!/bin/bash

#PBS -P vp91
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=192gb
#PBS -l walltime=00:02:00
#PBS -l storage=scratch/vp91
#PBS -l jobfs=190GB
#PBS -e test.err
#PBS -o test.out
### set email notification
#PBS -m bea
#PBS -M frederick.fung@anu.edu.au
#PBS -l wd

#module load openmpi/4.0.7
module load nvidia-hpc-sdk/22.5  
#export OMP_NUM_THREADS=6

### launch the application
nsys profile --trace=openacc,nvtx   --force-overwrite true -o monte-carlo-gpu  ./monte-carlo-pi-openacc
#./monte-carlo-pi-openmp
