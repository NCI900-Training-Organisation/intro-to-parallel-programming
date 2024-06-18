import numpy as np
from numba import vectorize, float64

import time
from codetiming import Timer

@Timer(name="without_vectorization", text="CPU time (without_vectorization): {milliseconds:.0f} ms")
def without_vectorization(a: float, b: float) -> float:
    return np.cos(np.sqrt(a**2 + b**2) + 100 + np.sin(b))

@Timer(name="with_vectorization", text="CPU time (with_vectorization): {milliseconds:.0f} ms")
@vectorize([float64(float64, float64)]) 
def with_vectorization(a: float, b: float) -> float:
    return np.cos(np.sqrt(a**2 + b**2) + 100 + np.sin(b))

a = np.array([n for n in range(10_000_000)])
b = np.array([n for n in range(10_000_000)])

without_vectorization(a, b)

# In Numba the first call will not give good performance
# This is because, first call always involves code transformations
with_vectorization(a, b)
with_vectorization(a, b)