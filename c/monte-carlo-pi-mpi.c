/* =================================================================
MC_pi.c

Pi value approximation by Monte Carlo method.

Compilation:  refer to the Makefile

Usage:  mpirun -np 4 ./MC_pi

Prepared for NCI Training. 

Frederick Fung 2022
4527FD1D

Please leave comments at frederick.fung@anu.edu.au
====================================================================*/
#include<stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<mpi.h>


#define N (1000000<<2)
#define MATH_PI acos(-1.0)
#define RAND time(NULL)

int main(int argc, char *argv[]){

MPI_Init(&argc, &argv);

int rank, size;
double x, y, start_t, end_t;

start_t = MPI_Wtime();
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

int count=0;
int count_tot;
unsigned int seed ;

/* distribute number of sample points to processes */
int start = rank * N / size;
int end = (rank+1) * N /size;
printf("start %d end %d rank %d \n", start, end , rank);

/* generate seed for rand_r */
seed = rank +RAND;
for (int i=start; i<end; i++){
    x = rand_r(&seed)/ (double) RAND_MAX;
    y = rand_r(&seed)/ (double) RAND_MAX;

    if (x*x + y*y <= 1.0f)  count+=1;
}

/* collect results from all processes to root process 0 */
MPI_Reduce(&count, &count_tot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

if (rank ==0){
    printf("Hit numbers  %d Approx Pi %f\n", count_tot, (double)count_tot / N *4.0 );
}

end_t = MPI_Wtime();

 /* optional,for ordered prints only */ 
MPI_Barrier(MPI_COMM_WORLD); 
printf("MPI program runtime = %f on rank %d\n", end_t-start_t, rank);
MPI_Finalize();
}
