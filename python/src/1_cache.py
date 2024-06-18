from pypapi import papi_low as papi
from pypapi import events

import random
import numpy as np
import sys
import time
from codetiming import Timer

@Timer(name="simpleSum", text="CPU time: {milliseconds:.0f} ms")
def simpleSum(arrA, ArrB):
    np.add(arrA, ArrB)

papi.library_init()

evs = papi.create_eventset()
papi.add_event(evs, events.PAPI_L3_TCM)

papi.start(evs)
arr1 = np.array([n for n in range(10)])
arr2 = np.array([n for n in range(10)])
simpleSum(arr1, arr2)
result = papi.stop(evs)
print("Data cache miss (first-call): ", end="")
print(result)

papi.start(evs)
simpleSum(arr1, arr2)
result = papi.stop(evs)
print("Data cache miss (second-call): ", end="")
print(result)


papi.start(evs)
arr1 = np.array([n for n in range(10_000_000)])
arr2 = np.array([n for n in range(10_000_000)])
simpleSum(arr1, arr2)
result = papi.stop(evs)
print("Data cache miss (third-call): ", end="")
print(result)

papi.start(evs)
arr1 = np.array([n for n in range(20_000_000)])
arr2 = np.array([n for n in range(20_000_000)])
simpleSum(arr1, arr2)
result = papi.stop(evs)
print("Data cache miss (fourth-call): ", end="")
print(result)

papi.cleanup_eventset(evs)
papi.destroy_eventset(evs)