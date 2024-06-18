/* =================================================================
monte-carlo-pi-openacc.c

Written by Frederick Fung for  NCI-UNSW Parallel Programing Workshop Mar 2023

This program approximates the pi value by Monte-Carlo method. 

The code is accelerated by openacc multi-threading. 

Compilation: refer to the Makefile

Usage ./monte-carlo-pi-openacc

.....................................................................

Copyright under CC by Frederick Fung 2022

====================================================================*/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>
#include<math.h>
#include <nvToolsExt.h> 

#define MATH_PI acos(-1.0)
#define RAND time(NULL)
#define NVTX

int calc_pi(int samples, double *random_array){
    
    double x ;
    double y ;
    int count =0;
    int j=0;

    #pragma acc parallel loop private(x,y,j) reduction(+:count) 
    for (j=0; j<samples; j++){
             x = random_array[j];
             y = random_array[2*j];
            if (x*x + y*y <= 1.0)  {count+=1;}
    }
    return count;
}

int main(int argc, char** argv)
{   
    int trials[]={1000000<<2,1000000<<3, 1000000<<4, 1000000<<5, 1000000<<6, 1000000<<7, 1000000<<8, 1000000<<9};
    printf("MATH Pi %f\n", MATH_PI);
    for (int i = 0; i< sizeof(trials) / sizeof(trials[0]); i++){
        
        #ifdef NVTX
        nvtxRangePush("MemAllocTime");
        #endif
        
        double start = omp_get_wtime();
        
        /* assemble the random number array at CPU */
        unsigned int seed = RAND;
        int samples = trials[i];
        double *random_array = malloc(2* samples * sizeof(double));
        for (int k=0; k< 2*samples; k++){
            random_array[k]= rand_r(&seed)/ (double) RAND_MAX;
        }
        
        #ifdef NVTX
        nvtxRangePop();
        #endif
        
        int hit;
        
        /* offload the compuation to GPU */
        hit = calc_pi(samples, random_array);
        
        double end = omp_get_wtime();
        printf("Sampling points %d; Hit numbers %d; Approx Pi %f, Total time in %f seconds \n", samples, hit, (double) hit/ samples * 4.0f, end -start);
        free(random_array);
    }
    return 0;
}
