from mpi4py import MPI
import cupy as cp


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = cp.arange(5, dtype='i')
recvbuf = cp.empty_like(sendbuf)
assert hasattr(sendbuf, '__cuda_array_interface__')
assert hasattr(recvbuf, '__cuda_array_interface__')

cp.cuda.get_current_stream().synchronize()

print("sendbuf on process " + str(rank)+ ": ", end=" ")
print(sendbuf)

comm.Allreduce(sendbuf, recvbuf)

print("recvbuf on process " + str(rank)+ ": ", end=" ")
print(recvbuf)

assert cp.allclose(recvbuf, sendbuf*size)
