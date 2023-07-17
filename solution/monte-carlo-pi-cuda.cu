/* =================================================================
monte-carlo-pi-cuda.c

Written by Frederick Fung for NCI-Monash Parallel Programing Workshop July 2023

This program approximates the pi value by Monte-Carlo method. 

The code is accelerated by openaccand cuda multi-threading. 

Compilation: refer to the Makefile

Usage ./monte-carlo-pi-openacc

.....................................................................

Copyright under CC by Frederick Fung 2023

====================================================================*/

#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define cudaCheck(expr) \
	do {\
		cudaError_t e = (expr); \
		if (e != cudaSuccess) { \
			fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
			abort(); \
		} \
	} while(false)


#define MATH_PI acos(-1.0)

#define NUM_BLOCKS 256
#define NUM_THREADS 256


__global__ void calc_pi(size_t *d_count, size_t samples) {
	// Define some shared memory: all threads in this block
	__shared__ size_t counter[NUM_THREADS];

	// Unique ID of the thread
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Initialize RNG
	curandState_t rng;
	curand_init(clock64(), tid, 0, &rng);

	// Initialize the counter
	//counter[threadIdx.x] = 0;

	// Computation loop
	//for (int i = 0; i < samples; i++) {
		float x = curand_uniform(&rng); // Random x position in [0,1]
		float y = curand_uniform(&rng); // Random y position in [0,1]
		counter[threadIdx.x] = 1 - int(x * x + y * y); // 
	//}

    __syncthreads();

	// The first thread in *every block* should sum the results
	if (threadIdx.x == 0) {
		// Reset count for this block
		size_t block_count = 0;
		// Accumulate results
		for (int i = 0; i < NUM_THREADS; i++) {
			block_count += counter[i];
		}
		d_count[blockIdx.x] = block_count;
	}
}

int main()
{
    size_t trials[] = {1000000<<2,1000000<<3, 1000000<<4, 1000000<<5, 1000000<<6, 1000000<<7, 1000000<<8, 1000000<<9};
    int num_trials = sizeof(trials) / sizeof(trials[0]);

    printf("MATH Pi %f\n", MATH_PI);
    printf("/////////////////////////////////////////////////////\n" );

    //curandState* states;
    //cudaMalloc((void**) &states, NUM_BLOCKS * NUM_THREADS * sizeof(curandState));



    for (int i = 0; i < num_trials; ++i) {
        size_t samples = trials[i];
        size_t *h_count= (typeof(h_count))malloc(sizeof(*h_count) * NUM_BLOCKS);
        size_t *d_count;

        cudaCheck(cudaMalloc(&d_count, sizeof(*d_count) * NUM_BLOCKS)); // Device memory
        //cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

      
        calc_pi<<<NUM_BLOCKS, NUM_THREADS>>>(d_count, samples );
        cudaCheck(cudaGetLastError());
		cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(h_count, d_count, sizeof(*h_count) * NUM_BLOCKS, cudaMemcpyDeviceToHost));

		size_t count = 0;
		for (int j = 0; j < NUM_BLOCKS; ++j) count += h_count[j];

        double pi_approx = (double) count / trials[i] * 4.0f;
        printf("Sampling points %lu; Hit numbers %lu; Approx Pi %g\n", trials[i], count, pi_approx);

        cudaFree(d_count);
		free(h_count);
    }

    //cudaFree(states);

    return 0;
}
