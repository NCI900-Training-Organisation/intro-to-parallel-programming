HPC Compute Node
****************

.. image::  ../figs/computeNode.drawio.png

A cache is a small, high-speed storage component used to temporarily hold frequently accessed data or instructions
(`temporal locality <https://www.sciencedirect.com/topics/computer-science/temporal-locality>`_ ) to improve system performance. Its primary purpose is to 
reduce the time it takes for a processor to access data from the main memory (RAM) or other slower storage devices.



L1, L2, and L3 caches are hierarchical levels of CPU cache designed to speed up data access and improve overall processing performance:

- **L1 Cache**: This is the smallest and fastest cache level, located directly on the CPU chip. It typically includes separate caches for instructions (L1i) and data (L1d). Due to its proximity to the CPU cores, it provides the quickest access to frequently used data and instructions, but it has limited capacity.

- **L2 Cache**: Larger than L1 but slower, the L2 cache is also located on the CPU chip or very close to it. It serves as an intermediary between the fast L1 cache and the slower L3 cache or main memory. It holds data and instructions that are not immediately needed by L1 but are accessed frequently enough to justify faster access than the main memory.

- **L3 Cache**: This is the largest and slowest of the three caches, typically shared among multiple CPU cores. It acts as a last-level cache before data is fetched from main memory. The L3 cache improves performance by storing a larger amount of data that is likely to be used by multiple cores, thus reducing the number of memory accesses and potential bottlenecks.

Together, these cache levels balance speed and capacity to enhance CPU performance by minimizing data access times.

How does cache influence peformance?
************************************

In the context of caching, **cache hit** and **cache miss** refer to the outcomes of a cache lookup operation:

- **Cache Hit**: A cache hit occurs when the data or instruction requested by the CPU is found in the cache. This means the cache contains a copy of the data that is needed, allowing the CPU to access it quickly and avoid fetching it from the slower main memory. Cache hits improve performance by reducing access time and latency.

- **Cache Miss**: A cache miss happens when the requested data or instruction is not found in the cache. In this case, the system must retrieve the data from the main memory or another slower storage medium. After fetching the data, it is typically stored in the cache for future use. Cache misses can result in slower access times since the data must be retrieved from a less efficient source.

Overall, maximizing cache hits and minimizing cache misses are key strategies for optimizing system performance and efficiency. Also, as the data size increases, 
cache misses also increase, leading to performance degradation.

.. code-block:: console
    :linenos:
    
    qsub 1_cachePapi.pbs

Are you getting linear peformance for third and fourth call?