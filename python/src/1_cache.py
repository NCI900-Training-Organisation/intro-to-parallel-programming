from pypapi import papi_low as papi
from pypapi import events

import random
import numpy as np

papi.library_init()

evs = papi.create_eventset()
papi.add_event(evs, events.PAPI_L1_DCM)

papi.start(evs)

arr1 = np.array([3, 2, 1])
arr2 = np.array([1, 2, 3])
out_arr = np.add(arr1, arr2)

result = papi.stop(evs)
print(result)

papi.start(evs)

out_arr = np.add(arr1, arr2)

result = papi.stop(evs)

print(result)



papi.cleanup_eventset(evs)
papi.destroy_eventset(evs)