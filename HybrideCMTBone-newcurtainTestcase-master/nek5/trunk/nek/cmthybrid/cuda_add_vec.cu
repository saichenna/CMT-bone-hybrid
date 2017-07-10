#include "cuda_add_vec.h"

__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
         
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

__global__ void vecMul(double *a, double*b, int jobs, int n){
	// Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
         
    // Make sure we do not go out of bounds
    	if (id < n){
		for(int i = 0; i< jobs; i++)
        		b[i*n+id] = b[i*n+id] * a[i*n+id];
	}

}

__global__ void vecCopy(double *a, double*b, int jobs, int n){
	// Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
         
    // Make sure we do not go out of bounds
    if (id < n){
        for(int i = 0; i< jobs; i++)
            b[i*n+id] = a[i*n+id];
    }

}

