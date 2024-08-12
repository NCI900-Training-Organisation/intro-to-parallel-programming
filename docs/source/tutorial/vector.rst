Vector Parallelism
------------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 20 min
    * **Exercises:** 5 min

        **Objectives:**
            #. Learn about vector parallelism.

Vector parallelism is a form of parallel computing that leverages the simultaneous processing of multiple data 
elements using vector processors or SIMD (Single Instruction, Multiple Data) instructions. 

.. image::  ../figs/vectorPrallelism.drawio.png


How does vectorization influence peformance?
*******************************************

Vector processors are specialized hardware designed to handle vector operations. Instead of processing single 
data elements sequentially, they can operate on entire vectors (arrays of data) in parallel. The effectiveness 
of vector parallelism depends on the *vector length*, which is the number of data elements a vector processor 
can handle in parallel.

We will use **Numba** to vectorrize python code.

.. code-block:: console
    :linenos:
    qsub 2_vectorize.pbs


.. admonition:: Key Points
   :class: hint

    #. Vector parallelism can improve the peformance of the application.
    #. Vector length decides the limit of vector parallelism.
    #. Numba provides an easy way to vectorize code in Python.