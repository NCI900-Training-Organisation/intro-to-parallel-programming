/* =================================================================
monte-carlo-pi-openacc.c

Written by Frederick Fung for NCI-Monash Parallel Programing Workshop Jul 2023

This program approximates the pi value by Monte-Carlo method. 

The code is accelerated by openacc multi-threading. 

Compilation: refer to the Makefile

Usage ./monte-carlo-pi-openacc

.....................................................................

Copyright under CC by Frederick Fung 2023

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

void calc_pi(int *hit, int *samples ){

    double x = 0;
    double y = 0;
    long i;
    int count =0;
    
    unsigned int seed = omp_get_thread_num()+RAND;
    #pragma acc
    for (i=0; i<*samples; i++){
            x = rand_r(&seed)/ (double) RAND_MAX;
            y = rand_r(&seed)/ (double) RAND_MAX;
            if (x*x + y*y <= 1.0f)  count+=1;
        }  
     *hit = count;
}

int main ()
{   int trials[]={1000000<<2,1000000<<3, 1000000<<4, 1000000<<5, 1000000<<6, 1000000<<7, 1000000<<8};

    printf("MATH Pi %f\n", MATH_PI);
    printf("/////////////////////////////////////////////////////\n" );
    for (int i = 0; i< sizeof(trials) / sizeof(trials[0]); i++){
        
        int samples = trials[i];
        int hit;
        double start = omp_get_wtime();
        hit = 0;
        calc_pi(&hit, &samples);
        double end = omp_get_wtime();
        printf("Sampling points %d; Hit numbers %d; Approx Pi %f, Total time in %f seconds \n", samples, hit, (double) hit/ samples * 4.0f, end -start);
    }
    return 0;
}