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

Python Packages (in order of installation):
1. python-papi
2. numpy


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
qsub 1_testScript.pbs
```



## HPC Compute Node

![](figs/computeNode.drawio.png)

```
qsub 2_cachePapi.pbs
```

## How does cache influence peformance?

## Vector Parallelism

![](figs/vectorPrallelism.drawio.png)

## Multi-core Parallelism

![](figs/multicorePrallelism.drawio.png)


## Multi-node Parallelism

![](figs/multinodePrallelism.drawio.png)

