import numpy as np
from numba import vectorize, float64

import time
from codetiming import Timer

@Timer(name="without_vectorization", text="CPU time (without vectorization): {milliseconds:.0f} ms")
def without_vectorization(a: float, b: float) -> float:
    return np.add(a, b) 

@Timer(name="with_vectorization", text="CPU time (with vectorization): {milliseconds:.0f} ms")
@vectorize([float64(float64, float64)]) 
def with_vectorization(a: float, b: float) -> float:
    return np.add(a, b) 

N = 4_000_000
a = np.array([n for n in range(N)])
b = np.array([n for n in range(N)])

without_vectorization(a, b)

# In Numba the first call will not give good performance
# This is because, first call always involves code transformations
with_vectorization(a, b)
with_vectorization(a, b)