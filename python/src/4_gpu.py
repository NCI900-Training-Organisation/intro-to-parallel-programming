import numpy as np
import numba
from numba import njit, prange

import time
from codetiming import Timer



@Timer(name="without_parallelization", text="CPU time (without parallelization): {milliseconds:.0f} ms")
def without_parallelization(a: float, b: float) -> float:
    return np.cos(np.sqrt(a**2 + b**2) + 100 + np.sin(b))

@Timer(name="with_parallelization", text="CPU time (with parallelization): {milliseconds:.0f} ms")
@numba.jit(nopython=True, parallel=True)
def with_parallelization(a: float, b: float) -> float:
    return np.cos(np.sqrt(a**2 + b**2) + 100 + np.sin(b))

a = np.array([n for n in range(10_000_000)])
b = np.array([n for n in range(10_000_000)])

without_parallelization(a, b)

# In Numba the first call will not give good performance
# This is because, first call always involves code transformations
with_parallelization(a, b)
with_parallelization(a, b)