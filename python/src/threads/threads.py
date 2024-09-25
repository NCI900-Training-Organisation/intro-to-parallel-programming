import logging
import threading
import time
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

sum  = 0
lock = threading.Lock()

def thread_kernel_add(thread_index, repeat, value):
    logging.info("I am thread %s", thread_index)

    global sum # modify the scope of the variable
    global lock
    logging.info("Initial sum in adder thread %s : %d", thread_index, sum)
   
    lock.acquire()
    for i in range(repeat):
        tmp = sum
        sleep(0)
        tmp = tmp + value
        sleep(0)
        sum = tmp
    lock.release()

    logging.info("Final sum in adder thread %s : %d", thread_index, sum)


def thread_kernel_sub(thread_index, repeat, value):
    logging.info("I am thread %s", thread_index)

    global sum # modify the scope of the variable
    global lock
    logging.info("Initial sum in subtractor thread %s : %d", thread_index, sum)
    
    lock.acquire()
    for i in range(repeat):
        tmp = sum
        sleep(0)
        tmp = tmp - value
        sleep(0)
        sum = tmp
    lock.release()

    logging.info("Final sum in subtractor thread %s : %d", thread_index, sum)


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")  
    
    lock = threading.Lock()
    
    # launch the threads
    adder = threading.Thread(target=thread_kernel_add, args=(1, 1000000, 100) ) # adder threads
    adder.start()
    
    subtractor = threading.Thread(target=thread_kernel_sub, args=(2, 1000000, 100) ) # subtractor threads
    subtractor.start()
    
    adder.join()
    subtractor.join()
    
    print("Final sum = "+ str(sum))
