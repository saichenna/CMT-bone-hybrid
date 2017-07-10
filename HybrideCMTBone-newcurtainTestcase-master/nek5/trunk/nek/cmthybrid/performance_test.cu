// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "nvml.h"
#include "performance_test.h"

// includes, project
//#include "magma.h"
#include "cuda_multi_gemm_unif.cu"
//#include "cuda_add_vec.h"

//My includes
#include "debug_fns.h"
#include "transformations.h"

//switch the comments to toggle debug mode
//#define D 
#define D for(;0;)

double get_time( void )
{
    struct timeval t;
    gettimeofday( &t, NULL );
    return t.tv_sec + t.tv_usec*1e-6;
}

__global__ void vecCopy(double *a, double*b, int jobs, int n){
	// Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
         
    // Make sure we do not go out of bounds
    /*if (id < n){
        for(int i = 0; i< jobs; i++)
            b[i*n+id] = a[i*n+id];
    }*/
    if(id < n*jobs)
        b[id] = a[id];

}
__global__ void vecMul(double *a, double*b, int jobs, int n){
	// Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
         
    // Make sure we do not go out of bounds
    /*if (id < n){
	for(int i = 0; i< jobs; i++)
            b[i*n+id] = b[i*n+id] * a[i*n+id];
    }*/
    if(id < n*jobs)
        b[id] = b[id] * a[id];

}


__global__ void full2face(double *vols, double*faces, int nel, int n, int nxyz, int*iface){

    //6 faces, each of size nx * nz => n = nx*nz *6
    //vols: e elements each of size nx*ny*nz => nxyz = nx*ny*nz

    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<nel*n){//n = nxyz
        int e = id/n; //+1 in fortran
        int j = id%n; //+1 in fortran
        int i = iface[id];//[e][j];
        faces[id]/*[e][j]*/ = vols[e*nxyz+i]/*[e][i]*/;
    }

    
}

__global__ void face2full(double *vols, double*faces, int nel, int n, int nxyz, int*iface){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<nel*n){//n = nxyz
        int e = id/n; //+1 in fortran
        int j = id%n; //+1 in fortran
        int i = iface[id];//[e][j]
        vols[e*nxyz+i] = vols[e*nxyz+i] + faces[id];
    }

}

void full2faceWrapper_(double *vols, double*faces, int nel, int n, int nxyz, int*iface, bool device_arr, bool pull_result){
    // n = nx * nz
    // Device input arrays
    double *d_vols;
    double *d_faces;
    int *d_iface;
    // allocate device vectors memory
    cudaMalloc(&d_vols, nxyz*nel*sizeof(double));
    cudaMalloc(&d_faces, n*nel*sizeof(double));
    cudaMalloc(&d_iface, n*nel*sizeof(int));
    cudaMemcpy( d_vols, vols, nxyz*nel*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iface, iface, n*nel*sizeof(int), cudaMemcpyHostToDevice);
    
    int blockSize, gridSize;
     
    // Number of threads in each thread block
    blockSize = 1024;
              
    // Number of thread blocks in grid
    //gridSize = (int)ceil((float)n*k/blockSize);
    gridSize = (int)ceil((float)n*nel/blockSize);

    // Execute the kernel
    full2face<<<gridSize, blockSize>>>(d_vols, d_faces, nel, n, nxyz, d_iface);
    cudaMemcpy( faces, d_faces, n*nel*sizeof(double), cudaMemcpyDeviceToHost );
    // Release device memory
    cudaFree(d_faces);
    cudaFree(d_vols);
    cudaFree(d_iface);

}


void init_matrix(double * mat, int size, int begin){
        for(int i=0; i<size; i++){
                mat[i] = begin+i;
                // mat[i] = rand();
        }
}

void init_u(double * mat, int n, int k, int jobs){

    //for(int i=0; i<n*k*jobs;i++)
        //mat[i] = 0.0;

    size_t bytes = jobs*n*k*sizeof(double);

    double * u_eq1; //jobs(number of elements) * k
    cudaMallocHost( (void**) &u_eq1, bytes);
    init_matrix(u_eq1, jobs*n*k, 10);
    
    double * vx; //working only on vx direction
    cudaMallocHost( (void**) &vx, bytes);
    init_matrix(vx, jobs*n*k, 10);
    
    //calc time
    float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);


    // Device input vectors
    double *d_mat;
    double *d_u_eq1;
    double *d_vx;
    // allocate device vectors memory
    cudaMalloc(&d_mat, bytes);
    cudaMalloc(&d_u_eq1, bytes);
    cudaMalloc(&d_vx, bytes);
    // copy host vectors to device
    cudaMemcpy( d_u_eq1, u_eq1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_vx, vx, bytes, cudaMemcpyHostToDevice);
    
    int blockSize, gridSize;
     
    // Number of threads in each thread block
    blockSize = 1024;
              
    // Number of thread blocks in grid
    //gridSize = (int)ceil((float)n*k/blockSize);
    gridSize = (int)ceil((float)n*k*jobs/blockSize);

    // Execute the kernel
    vecCopy<<<gridSize, blockSize>>>(d_u_eq1, d_mat, jobs, n*k);
    vecMul<<<gridSize, blockSize>>>(d_vx, d_mat, jobs, n*k);
                          
    // Copy array back to host
    cudaMemcpy( mat, d_mat, bytes, cudaMemcpyDeviceToHost );
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("second element is %f, n %d, k%d, time is %f\n",mat[n*k+1],n,k,time*1e-03);
                                         


    //do in cpu
    cudaEventRecord(startEvent, 0);
    for(int i =0; i< n*k*jobs; i++)
        mat[i] = u_eq1[i];
    for(int i=0; i< n*k*jobs;i++)
        mat[i] = mat[i] * vx[i];
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("cpu time is %f\n",time*1e-03);

    /*int nxz= 6, nxyz = 24, nel=3000;
    double *vols = new double[nxyz*nel];
    double *faces = new double[nxz*nel];
    int *iface = new int[nxz*nel];
    vols[1*nxyz+12] = 2.3;
    iface[1*nxz+2] = 12;
    full2faceWrapper_(vols, faces, nel, nxz, nxyz, iface, true, true);
    printf("face = %f\n",faces[1*nxz+2]);*/
}

//program
extern "C" void test_( int* matsize_p, int* gridsize_p, int* jobs_p, double* h_A,
            double* h_AA, int* M_p, int* N_p, int* K_p)
{
        int matsize = *matsize_p;
        int gridsize = *gridsize_p;
        int jobs = *jobs_p;
        int M = *M_p;
        int N = *N_p;
        int K = *K_p;
	float time;
        cudaEvent_t startEvent, stopEvent;

	cudaDeviceProp prop;
	cudaGetDeviceProperties (&prop, 0);

	cudaSetDevice( 0 );

        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

	double *h_B, *h_BB, *h_C, *h_D, *h_E;
	double *d_A, *d_AA, *d_B, *d_BB; 
	double *d_C, *d_D, *d_E;

	M = matsize;
	N = matsize*matsize;
	K = matsize;

	cudaMallocHost( (void**) &h_B, (K*N)*sizeof(double)*jobs );
	cudaMallocHost( (void**) &h_BB, (N*K)*sizeof(double)*jobs );
	cudaMallocHost( (void**) &h_C, (K*N)*sizeof(double)*jobs );
	cudaMallocHost( (void**) &h_D, (K*N)*sizeof(double)*jobs );
	cudaMallocHost( (void**) &h_E, (N*K)*sizeof(double)*jobs );

        /* Initialize and copy the matrices */
        //init_matrix(h_B, N*K*jobs, 10);
        init_u(h_B, N, K, jobs);
        // memset(h_C, 0, (K*N)*sizeof(double)*jobs);
        // memset(h_D, 0, (K*N)*sizeof(double)*jobs);
        // memset(h_E, 0, (K*N)*sizeof(double)*jobs);

	cudaMalloc( (void**) &d_A, (M*K)*sizeof(double) );
	cudaMalloc( (void**) &d_AA, (K*M)*sizeof(double) );
	cudaMalloc( (void**) &d_B, (K*N)*sizeof(double)*jobs );
	cudaMalloc( (void**) &d_BB, (N*K)*sizeof(double)*jobs );
	cudaMalloc( (void**) &d_C, (K*N)*sizeof(double)*jobs );
	cudaMalloc( (void**) &d_D, (K*N)*sizeof(double)*jobs );
	cudaMalloc( (void**) &d_E, (N*K)*sizeof(double)*jobs );

        // cudaMemset(d_C, 0, (K*N)*sizeof(double)*jobs);
        // cudaMemset(d_D, 0, (K*N)*sizeof(double)*jobs);
        // cudaMemset(d_E, 0, (K*N)*sizeof(double)*jobs);

	cudaStream_t stream;
	cudaStreamCreate( &stream );

	D printf("Matrix d:\n");
	D print(h_A, M, K);

	D printf("Matrix db:\n");
	D print(h_AA, M, K);

	D printf("Matrix u:\n");
	D print(h_B, K, N);

	
	D printf("Matrix ub:\n");
	D print(h_BB, K, N);

	const double alpha = 1;
	const double beta = 0;
	unsigned int dim = K;

	

	cublasSetMatrix(M, K, sizeof(double), h_A, K, d_A, K);
	cublasSetMatrix(K, M, sizeof(double), h_AA, K, d_AA, K);

        cudaEventRecord(startEvent, 0);

	cublasSetMatrixAsync(M, N*jobs, sizeof(double), h_B, M, d_B, M, stream);
	fflush( stdout );

	cuda_multi_gemm_unif(stream, 'N', 'N', dim, dim, dim, &alpha, dim, dim*dim, d_A, d_B, d_BB, &beta, d_C, d_D, d_E, jobs*K, gridsize);
	cudaDeviceSynchronize();

	fflush( stdout );

	cublasGetMatrixAsync(M, N*jobs, sizeof(double), d_C, K, h_C, K, stream);
	cublasGetMatrixAsync(M, N*jobs, sizeof(double), d_D, K, h_D, K, stream);
	cublasGetMatrixAsync(M, N*jobs, sizeof(double), d_E, K, h_E, K, stream);

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&time, startEvent, stopEvent);

        // printf("GPU time: %f, throughput: %f\n", time * 1e-03, (jobs*2.0*3*K*K*K*K)/(1024*1024*1024*time*1e-03));
        printf(" gpu time: %f\n", time * 1e-03);

	D printf("Matrix r:\n"); 
	D print((h_C), M, N);

	D printf("Matrix s:\n"); 
	D print((h_D), M, N);

	D printf("Matrix t:\n"); 
	D print((h_E), M, N);

	cudaFreeHost( h_B );
	cudaFreeHost( h_BB );
	cudaFreeHost( h_C );
	cudaFreeHost( h_D );
	cudaFreeHost( h_E );
            
	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );

	fflush( stdout );

        /**jobs_p = 22;
        int **p1 = new int*[2];
        for(int i=0; i<2;i++)
            p1[i] = new int[3];
        int *p2 = (int*)p1;
        for(int i=0;i<2;i++)
            for(int j=0;j<3;j++){
                p1[i][j] = i*2+j;
                printf("a[%d][%d]=%d,%d\n",i,j,p1[i][j],p2[i*2+j]);
            }*/
	return;

}

