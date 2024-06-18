# Introduction to Parallel Programming Using Python
This repository provides an introduction to the concepts of parallel programming using Python.

Learning outcomes of the tutorial are:
1. Learn the basics concpets of parallel programming.
2. Learn the different harware components that make up a HPC machine. 
3. Learn how an HPC machine is organized.
4. Lear how to submit a Job to an PBS batch scheduler. 

Prerequisite:
1. Experience with Python.
2. Experience with bash or similar unix shells.

Modules:
1. python3/3.11.0
2. papi/7.0.1
3. openmpi/4.1.4

Python Packages (in order of installation):
1. python-papi
2. numpy
3. codetiming
4. numba
5. mpi4py


## High-level HPC Architecture

![](figs/HPC_overview.drawio.png)

### Requesting a Job

1. Which project are you using?
2. Which job queue are you planning to use?
3. How many CPU cores are required for your task?
4. How many GPUs do you need?
5. What is the estimated runtime of your program?
6. Which modules are necessary to execute the program?
7. What script or command will you use to run the program?

```

#!/bin/bash

#PBS -P vp91
#PBS -q normal

#PBS -l ncpus=48
#PBS -l mem=10GB
#PBS -l walltime=00:02:00

#PBS -N testScript



module load python3/3.11.0
module load papi/7.0.1

. /scratch/vp91/Training-Venv/intro-parallel-prog/bin/activate

which python

```

```
cd python/jobScripts
qsub 0_testScript.pbs
```



## HPC Compute Node

![](figs/computeNode.drawio.png)

## How does cache influence peformance?

As the data size increases, cache misses also increase, leading to performance degradation.

```
qsub 1_cachePapi.pbs
```

Are you getting linear peformance for third and fourth call?



## Vector Parallelism

![](figs/vectorPrallelism.drawio.png)

### How does vectorization influence peformance?

We will use `Numba` to vectorrize python code.

```
qsub 2_vectorize.pbs
```

## Multi-core Parallelism

![](figs/multicorePrallelism.drawio.png)


## Multi-node Parallelism

![](figs/multinodePrallelism.drawio.png)

