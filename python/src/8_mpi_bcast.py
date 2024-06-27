from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.arange(5, dtype='i')
    print("Broadcast from process 0: ", end=" ")
    print(data)
else:
    data = np.empty(5, dtype='i')

comm.Bcast(data, root=0)

print("Broadcast received on process " + str(rank)+ ": ", end=" ")
print(data)
