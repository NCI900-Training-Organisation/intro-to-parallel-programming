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


#define MATH_PI acos(-1.0)

#define NUM_BLOCKS 256
#define NUM_THREADS 256


__global__ void calc_pi(int *d_count, long samples) {
	// Define some shared memory: all threads in this block
	__shared__ int counter[32];

	// Unique ID of the thread
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Initialize RNG
	curandState_t rng;
	curand_init(clock64(), tid, 0, &rng);

	// Initialize the counter
	counter[threadIdx.x] = 0;

	// Computation loop
	for (int i = 0; i < samples; i++) {
		float x = curand_uniform(&rng); // Random x position in [0,1]
		float y = curand_uniform(&rng); // Random y position in [0,1]
		counter[threadIdx.x] += 1 - int(x * x + y * y); // Hit test - I think this is clever- CA
	}

	// The first thread in *every block* should sum the results
	if (threadIdx.x == 0) {
		// Reset count for this block
		d_count[blockIdx.x] = 0;
		// Accumulate results
		for (int i = 0; i < 32; i++) {
			d_count[blockIdx.x] += counter[i];
		}
	}
}

int main()
{
    int trials[] = {1000000<<2,1000000<<3, 1000000<<4, 1000000<<5, 1000000<<6, 1000000<<7, 1000000<<8, 1000000<<9};
    int num_trials = sizeof(trials) / sizeof(trials[0]);

    printf("MATH Pi %f\n", MATH_PI);
    printf("/////////////////////////////////////////////////////\n" );

    curandState* states;
    cudaMalloc((void**) &states, NUM_BLOCKS * NUM_THREADS * sizeof(curandState));



    for (int i = 0; i < num_trials; ++i) {
        int samples = trials[i];
        int h_count=0;
        int *d_count;

        cudaMalloc(&d_count, sizeof(int) * NUM_BLOCKS); // Device memory
        //cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

      
        calc_pi<<<NUM_BLOCKS, NUM_THREADS>>>(d_count, samples );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        double pi_approx = (double) h_count / trials[i] * 4.0f;
        printf("Sampling points %d; Hit numbers %d; Approx Pi %f\n", trials[i], h_count, pi_approx);

        cudaFree(d_count);
    }

    cudaFree(states);

    return 0;
}
