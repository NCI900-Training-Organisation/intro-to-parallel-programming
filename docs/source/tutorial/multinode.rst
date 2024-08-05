Multi-node Parallelism
-----------------------

While all the aforementioned parallelism is beneficial, it is limited to a single node. To truly scale up an application, we need to use multiple nodes, i.e., distributed computing. The main challenge with distributed computing is that the memory in each node is distinct and separate, meaning there is no way for a thread in one node to access data in another node.

.. image::  figs/multinodePrallelism.drawio.png

We overcome this challenge by using message passing.

.. image::  figs/MPI.png

Broadcast Operation
*******************

.. image::  figs/bcast.png

GPU-aware MPI and All-Gather Operation
**************************************

.. image:: figs/allgather.png