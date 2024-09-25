import multiprocessing as mp
from multiprocessing import Process, Queue
import random


def rand_num(queue, lock):
    num = random.random()
    print("Process ", mp.current_process(), " : ",  num)

    with lock:
        queue.put(num)

if __name__ == "__main__":

    queue = Queue()
    lock = mp.Lock()
    processes = [Process(target=rand_num, args=(queue, lock, )) for x in range(2)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    results = [queue.get() for p in processes]
    print("Sum = ", sum(results))