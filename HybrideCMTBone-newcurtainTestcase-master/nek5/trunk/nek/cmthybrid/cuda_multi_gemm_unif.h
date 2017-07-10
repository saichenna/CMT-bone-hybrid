
#ifndef CUDA_MULTI_GEMM_UNIF_H
#define CUDA_MULTI_GEMM_UNIF_H

// -----------------------------------------------------------------------------

#include <cuda_runtime_api.h>

// -----------------------------------------------------------------------------

template<typename T>
bool cuda_multi_gemm_unif(
	cudaStream_t stream, char transa, char transb,
	unsigned int m, unsigned int n, unsigned int k,
	const T* alpha,
	const T* A, unsigned int lda, unsigned int lda2,
	const T* B, unsigned int ldb, unsigned int ldb2,
	const T* beta,
	T* C, unsigned int ldc, unsigned int ldc2,
	unsigned int batch_count,
	unsigned int grid_size);

// -----------------------------------------------------------------------------

#endif // CUDA_MULTI_GEMM_UNIF_H
