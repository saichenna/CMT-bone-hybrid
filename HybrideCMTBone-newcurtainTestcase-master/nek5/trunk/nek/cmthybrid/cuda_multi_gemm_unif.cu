
// -----------------------------------------------------------------------------

#include "cuda_multi_gemm_unif.h"
#include "integral_utils.h"
#include "static_assert.h"
#include "complex_utils.h"
#include <cuComplex.h>
#include <complex>
#include <cctype>
#include <cassert>

// -----------------------------------------------------------------------------

// m * m threads
template<typename T, unsigned int m, bool transa, bool transb, typename unary_a, typename unary_b, typename axpby_func_type>
__global__
void cuda_multi_gemm_unif_kernel_small(
	const T* A, unsigned int lda, unsigned int lda2,
	const T* B, unsigned int ldb, unsigned int ldb2,
	      T* C, unsigned int ldc, unsigned int ldc2,
	unsigned int batch_count,
	unary_a func_a, unary_b func_b, axpby_func_type axpby_func)
{
	const unsigned int n2pow = smallest_greater_or_equal_power_of_2<unsigned int, m>::value;

	__shared__ T Aloc[n2pow][n2pow + 1];
	__shared__ T Bloc[n2pow][n2pow + 1];

	const unsigned int tslow = threadIdx.x / m;
	const unsigned int tfast = threadIdx.x - m * tslow;
	const unsigned int off_a = tfast + lda * tslow;
	const unsigned int off_b = tfast + ldb * tslow;
	const unsigned int off_c = tfast + ldc * tslow;

	const unsigned int off2_a = lda2 * gridDim.x;
	const unsigned int off2_b = ldb2 * gridDim.x;
	const unsigned int off2_c = ldc2 * gridDim.x;

	A += lda2 * blockIdx.x;
	B += ldb2 * blockIdx.x;
	C += ldc2 * blockIdx.x;

	for(unsigned int mat_id = blockIdx.x; mat_id < batch_count; mat_id += gridDim.x)
	{
		__syncthreads();
		Aloc[transa ? tslow : tfast][transa ? tfast : tslow] = func_a(A[off_a]);
		Bloc[transb ? tfast : tslow][transb ? tslow : tfast] = func_b(B[off_b]);
		__syncthreads();

		A += off2_a;
		B += off2_b;
		T tmp = get_zero<T>();

		__syncthreads();
#pragma unroll
		for(unsigned int k = 0; k < m; ++k)
		{
			tmp += Aloc[tfast][k] * Bloc[tslow][k];
		}
		__syncthreads();

		axpby_func(C[off_c], tmp);
		C += off2_c;
	}
}

template<typename T, typename axpby_func_type>
bool cuda_multi_gemm_unif(
	axpby_func_type axpby_func,
	cudaStream_t stream, char transa, char transb,
	unsigned int m, unsigned int n, unsigned int k,
	const T* A, unsigned int lda, unsigned int lda2,
	const T* B, unsigned int ldb, unsigned int ldb2,
	T* C, unsigned int ldc, unsigned int ldc2,
	unsigned int batch_count,
	unsigned int grid_size)
{
	bool success =
		m == n &&
		n == k &&
		k <= 16;

	transa = tolower(transa);
	transb = tolower(transb);

#define trans_cases(m)  \
case m: \
	if(transa == 'n') { \
	if(transb == 'n') \
		cuda_multi_gemm_unif_kernel_small<T, m, false, false><<<grid_size, m*m, 0, stream>>> \
			(A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, identity_unary_func<T>(), identity_unary_func<T>(), axpby_func); \
	  else if(transb == 't') \
		cuda_multi_gemm_unif_kernel_small<T, m, false, true><<<grid_size, m*m, 0, stream>>> \
			(A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, identity_unary_func<T>(), identity_unary_func<T>(), axpby_func); \
	  else if(transb == 'c') \
		cuda_multi_gemm_unif_kernel_small<T, m, false, true><<<grid_size, m*m, 0, stream>>> \
			(A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, identity_unary_func<T>(), conjugate_unary_func<T>(), axpby_func); } \
	else if(transa == 't') { \
	if(transb == 'n') \
		cuda_multi_gemm_unif_kernel_small<T, m, true, false><<<grid_size, m*m, 0, stream>>> \
			(A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, identity_unary_func<T>(), identity_unary_func<T>(), axpby_func); \
	  else if(transb == 't') \
		cuda_multi_gemm_unif_kernel_small<T, m, true, true><<<grid_size, m*m, 0, stream>>> \
			(A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, identity_unary_func<T>(), identity_unary_func<T>(), axpby_func); \
	  else if(transb == 'c') \
		cuda_multi_gemm_unif_kernel_small<T, m, true, true><<<grid_size, m*m, 0, stream>>> \
			(A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, identity_unary_func<T>(), conjugate_unary_func<T>(), axpby_func); } \
	else if(transa == 'c') { \
	if(transb == 'n') \
		cuda_multi_gemm_unif_kernel_small<T, m, true, false><<<grid_size, m*m, 0, stream>>> \
			(A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, conjugate_unary_func<T>(), identity_unary_func<T>(), axpby_func); \
	  else if(transb == 't') \
		cuda_multi_gemm_unif_kernel_small<T, m, true, true><<<grid_size, m*m, 0, stream>>> \
			(A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, conjugate_unary_func<T>(), identity_unary_func<T>(), axpby_func); \
	  else if(transb == 'c') \
		cuda_multi_gemm_unif_kernel_small<T, m, true, true><<<grid_size, m*m, 0, stream>>> \
			(A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, conjugate_unary_func<T>(), conjugate_unary_func<T>(), axpby_func); } \
	break;

    switch(m)
	{
		trans_cases(1);
		trans_cases(2);
		trans_cases(3);
		trans_cases(4);
		trans_cases(5);
		trans_cases(6);
		trans_cases(7);
		trans_cases(8);
		trans_cases(9);
		trans_cases(10);
		trans_cases(11);
		trans_cases(12);
		trans_cases(13);
		trans_cases(14);
		trans_cases(15);
		trans_cases(16);
	}

	return success;
}

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
	unsigned int grid_size)
{
	// Fast code but makes nvcc slow

	//if(*alpha == get_a<T>(-1))
	//{
	//	if(	 *beta == get_a<T>(-1)) return cuda_multi_gemm_unif(axpby_ab<T, -1, -1>(), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	//	else if(*beta == get_a<T>(0))  return cuda_multi_gemm_unif(axpby_ab<T, -1,  0>(), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	//	else if(*beta == get_a<T>(1))  return cuda_multi_gemm_unif(axpby_ab<T, -1,  1>(), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);

	//	return cuda_multi_gemm_unif(axpby_a<T, -1>(*beta), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	//}
	//else if(*alpha == get_a<T>(0))
	//{
	//	if(	 *beta == get_a<T>(-1)) return cuda_multi_gemm_unif(axpby_ab<T, 0, -1>(), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	//	else if(*beta == get_a<T>(0))  return cuda_multi_gemm_unif(axpby_ab<T, 0,  0>(), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	//	else if(*beta == get_a<T>(1))  return cuda_multi_gemm_unif(axpby_ab<T, 0,  1>(), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);

	//	return cuda_multi_gemm_unif(axpby_a<T, 0>(*beta), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	//}
	//else if(*alpha == get_a<T>(1))
	//{
	//	if(	 *beta == get_a<T>(-1)) return cuda_multi_gemm_unif(axpby_ab<T, 1, -1>(), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	//	else if(*beta == get_a<T>(0))  return cuda_multi_gemm_unif(axpby_ab<T, 1,  0>(), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	//	else if(*beta == get_a<T>(1))  return cuda_multi_gemm_unif(axpby_ab<T, 1,  1>(), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);

	//	return cuda_multi_gemm_unif(axpby_a<T, 1>(*beta), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	//}
	//else
	//{
	//	if(	 *beta == get_a<T>(-1)) return cuda_multi_gemm_unif(axpby_b<T, -1>(*alpha), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	//	else if(*beta == get_a<T>(0))  return cuda_multi_gemm_unif(axpby_b<T,  0>(*alpha), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	//	else if(*beta == get_a<T>(1))  return cuda_multi_gemm_unif(axpby_b<T,  1>(*alpha), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);

	//	return cuda_multi_gemm_unif(axpby<T>(*alpha, *beta), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	//} 

	if(*alpha == get_a<T>(-1) && *beta == get_a<T>(0))
	{
		return cuda_multi_gemm_unif(axpby_ab<T, -1,  0>(), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	}
	else if(*alpha == get_a<T>(1) && *beta == get_a<T>(0))
	{
		return cuda_multi_gemm_unif(axpby_ab<T,  1,  0>(), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	}
	else if(*alpha == get_a<T>(1) && *beta == get_a<T>(1))
	{
		return cuda_multi_gemm_unif(axpby_ab<T,  1,  1>(), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	}
	else
	{
		return cuda_multi_gemm_unif(axpby<T>(*alpha, *beta), stream, transa, transb, m, n, k, A, lda, lda2, B, ldb, ldb2, C, ldc, ldc2, batch_count, grid_size);
	} 
}

// -----------------------------------------------------------------------------

#define CUDA_MULTI_GEMM_UNIF_ABC_INSTANTIATE(T)			\
template bool cuda_multi_gemm_unif<T>(					\
	cudaStream_t stream, char transa, char transb,		\
	unsigned int m, unsigned int n, unsigned int k,		\
	const T* alpha,										\
	const T* A, unsigned int lda, unsigned int lda2,	\
	const T* B, unsigned int ldb, unsigned int ldb2,	\
	const T* beta,										\
	T* C, unsigned int ldc, unsigned int ldc2,			\
	unsigned int batch_count,							\
	unsigned int grid_size)

CUDA_MULTI_GEMM_UNIF_ABC_INSTANTIATE(float);
CUDA_MULTI_GEMM_UNIF_ABC_INSTANTIATE(double);
CUDA_MULTI_GEMM_UNIF_ABC_INSTANTIATE(cuComplex);
CUDA_MULTI_GEMM_UNIF_ABC_INSTANTIATE(cuDoubleComplex);

// -----------------------------------------------------------------------------
