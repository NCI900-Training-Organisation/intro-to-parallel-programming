import numpy as np
import numba
from numba import njit, prange, cuda

import math

import time
from codetiming import Timer


@cuda.jit
def largeOp_using_cuda(an_array):

    gridDim_x, gridDim_y = cuda.grid(2)

    # blockId = (gridDim.x * blockIdx.y) + blockIdx.x
    blockId = (gridDim_x * cuda.blockIdx.y) + cuda.blockIdx.x

    # threadId = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x
    threadId = (blockId * (cuda.blockDim.x * cuda.blockDim.y)) + (cuda.threadIdx.x * cuda.blockDim.x) + cuda.threadIdx.x

    if threadId < an_array.size:  # Check array boundaries
        an_array[threadId]**2 + an_array[threadId]**2 * an_array[threadId]**2 / an_array[threadId]**2
        an_array[threadId]**2 + an_array[threadId]**2 * an_array[threadId]**2 / an_array[threadId]**2
        an_array[threadId]**2 + an_array[threadId]**2 * an_array[threadId]**2 / an_array[threadId]**2
        an_array[threadId]**2 + an_array[threadId]**2 * an_array[threadId]**2 / an_array[threadId]**2
        an_array[threadId]**2 + an_array[threadId]**2 * an_array[threadId]**2 / an_array[threadId]**2
        an_array[threadId]**2 + an_array[threadId]**2 * an_array[threadId]**2 / an_array[threadId]**2
        an_array[threadId]**2 + an_array[threadId]**2 * an_array[threadId]**2 / an_array[threadId]**2
        an_array[threadId]**2 + an_array[threadId]**2 * an_array[threadId]**2 / an_array[threadId]**2


@Timer(name="largeOp_using_numpy", text="CPU time without cuda: {milliseconds:.0f} ms")
def largeOp_using_numpy(an_array):  
    [(i**2 + i**2 * i**2 / i**2) for i in an_array]
    [(i**2 + i**2 * i**2 / i**2) for i in an_array]
    [(i**2 + i**2 * i**2 / i**2) for i in an_array]
    [(i**2 + i**2 * i**2 / i**2) for i in an_array]
    [(i**2 + i**2 * i**2 / i**2) for i in an_array]
    [(i**2 + i**2 * i**2 / i**2) for i in an_array]
    [(i**2 + i**2 * i**2 / i**2) for i in an_array]
    [(i**2 + i**2 * i**2 / i**2) for i in an_array]

arrayA = np.array([n for n in range(8_192_00)])

print("   ")
print("***** Large Computations *****")
largeOp_using_numpy(arrayA)

threadsperblock = (32, 32) # in V100 max threads ber block is 1024
blockspergrid_x = math.ceil(arrayA.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(arrayA.shape[0] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)



evtstart = numba.cuda.event(timing=True)
evtend = numba.cuda.event(timing=True)
stream = numba.cuda.stream()

evtstart.record()

arrayA_dev = numba.cuda.to_device(arrayA, stream=stream) # explicit data transfer
largeOp_using_cuda[blockspergrid, threadsperblock, stream](arrayA_dev)
arrayA_dev = arrayA_dev.copy_to_host()

evtend.record()
stream.synchronize()

elapsed_time = numba.cuda.event_elapsed_time(evtstart, evtend)

print("Kernel time with cuda: " + str(elapsed_time)+ " ms")

