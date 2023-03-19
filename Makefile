mc-serial: CC=gcc
mc-serial: CFLAGS=-g -Wall -fopenmp
mc-serial: CLIBS=-lm
mc-serial: monte-carlo-pi-serial.c
	${CC} ${CFLAGS} -o monte-carlo-pi-serial monte-carlo-pi-serial.c $(CLIBS)

mc-omp: CC=gcc 
mc-omp: CFLAGS=-g -fopenmp -Wall
mc-omp: CLIBS=-lm
mc-omp: monte-carlo-pi-openmp.c
	${CC} ${CFLAGS} -o monte-carlo-pi-openmp monte-carlo-pi-openmp.c $(CLIBS)

mc-acc: CC=nvc
mc-acc: CFLAGS=-g -Wall -acc -Minfo=accel -ta=tesla:managed 
mc-acc: CLIBS=-lm -lnvToolsExt
mc-acc: monte-carlo-pi-openacc.c
	${CC} ${CFLAGS} -o monte-carlo-pi-openacc monte-carlo-pi-openacc.c ${CLIBS}

mc-mpi: CC=mpicc
mc-mpi: CFLAGS=-g -Wall
mc-mpi: CLIBS=-lmpiP -lm -lbfd -liberty -lunwind
mc-mpi: monte-carlo-pi-mpi.c
	${CC} ${CFLAGS} -o monte-carlo-pi-mpi monte-carlo-pi-mpi.c ${CLIBS}
    
clean:
	rm -f *.o monte-carlo-pi-openacc monte-carlo-pi-openmp monte-carlo-pi-serial monte-carlo-pi-mpi
