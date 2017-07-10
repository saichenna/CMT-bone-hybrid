#ifndef CUDA_ADD_VEC_H
#define CUDA_ADD_VEC_H

// -----------------------------------------------------------------------------

#include <cuda_runtime_api.h>

// ----------------------------------------------------------------------------

void vecAdd(double *a, double *b, double *c, int n, int offset);
// n is the 3D array size, i.e. n = nx * ny * nz
// jobs is the number of elements e
__global__ void vecMul(double *a, double*b, int jobs, int n);
__global__ void vecCopy(double *a, double*b, int jobs, int n);

#endif
