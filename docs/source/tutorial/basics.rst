Basic of Parallelism
--------------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 15 min
    * **Exercises:** 10 min

        **Objectives:**
            #. Learn about the difference between threads and process

Process
********

A process is an instance of a program in execution. A process is responsible for executing a program's 
instructions and providing the environment in which the program operates (such as memory and I/O devices).

#. Program Code: The instructions of the program that the process is executing.
#. Process Stack: Contains temporary data such as method/function parameters, return addresses, and local variables.
#. Heap: A region of memory used for dynamic memory allocation during the process's execution.
#. Data Section: Contains global and static variables used by the process.
#. Process Control Block (PCB): A data structure maintained by the operating system that holds information about the process, including its state, program counter, CPU registers, and memory management information.

We can launch multiple process and the same time and the processes are isolated from each other.
Each process manages its own resources, including memory and CPU time and one application can
have more than one process. Process communicates between each other using Inter-Process Communication (IPC) 
mechanisms.

The OS can manage multiple process at the same time and the OS can switch the process executing in a CPU.
This involves saving the current state of the process that is being paused (the "old" process) and restoring 
the state of the process that is being resumed (the "new" process). The state of a process is 
captured in its *Process Control Block (PCB)*. This idea of switching between processes is called
*Context Switching*.

Threads
*******

A thread is the smallest unit of execution within a process. A process can contain multiple threads that 
share the same resources but execute independently. While each process is isolated from processes, threads 
within the same process share the same memory space and resources but they execute independently.

*Concurrency* refers to the ability to run multiple threads simultaneously. Threads can be managed by 
the operating system to run on different CPU cores, doing different computations, thereby 
improving performance. If two concurrent threads (or processes) can be run simultaneously we can say 
they are *parallel*.

Challenges with Threads
***********************

#. Synchronization: As threads share resources, they need mechanisms to synchronize access to prevent conflicts and ensure data consistency. Common synchronization tools include mutexes, semaphores, and locks.
#. Deadlock: A situation where two or more threads are waiting indefinitely for resources held by each other, leading to a standstill.
#. Race Conditions: Occur when the outcome depends on the unpredictable timing of thread execution, potentially causing inconsistent results.

.. admonition:: Key Points
   :class: hint

    #. Processes are isolated with separate memory spaces, while threads share the same memory space within a process.
    #. Processes have higher creation and management overhead due to separate resources and memory, whereas threads are lighter and cheaper to manage.
    #. Threads can communicate easily and efficiently since they share memory, while processes require more complex and resource-intensive Inter-Process Communication (IPC) mechanisms.




