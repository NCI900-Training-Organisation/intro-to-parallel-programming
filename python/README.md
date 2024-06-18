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

## High-level HPC Architecture

![](figs/HPC_overview.drawio.png)

### Requesting a Job



## HPC Compute Node

![](figs/computeNode.drawio.png)

## Vector Parallelism

![](figs/vectorPrallelism.drawio.png)

## Multi-core Parallelism

![](figs/multicorePrallelism.drawio.png)


## Multi-node Parallelism

![](figs/multinodePrallelism.drawio.png)





```

from dask_jobqueue import PBSCluster

```