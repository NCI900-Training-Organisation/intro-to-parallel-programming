import numpy as np
import numba
from numba import njit, prange, cuda

import math

import time
from codetiming import Timer


@Timer(name="increment_using_cuda", text="CPU time with cuda (launch time): {milliseconds:.0f} ms")
@cuda.jit
def increment_using_cuda(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1


@Timer(name="increment_using_numpy", text="CPU time without cuda: {milliseconds:.0f} ms")
def increment_using_numpy(an_array):
    new_array = li = [i+1 for i in an_array]


arrayA = np.array([n for n in range(10_000_000)])

increment_using_numpy(arrayA)

threadsperblock = (32, 32)
blockspergrid_x = math.ceil(arrayA.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(arrayA.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

evtstart = numba.cuda.event(timing=True)
increment_using_cuda[blockspergrid, threadsperblock](arrayA)
evtend = numba.cuda.event(timing=True)
elapsed_time = numba.cuda.event_elapsed_time(evtstart, evtend)

print("Time with cuda: " + str(elapsed_time))