//ll includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "nvml.h"
#include "cuda_functions.h"

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
        //int j = id%n; //+1 in fortran
        int i = iface[id];//[e][j];
        faces[id]/*[e][j]*/ = vols[e*nxyz+i-1]/*[e][i]*/;
        //faces[id] = 2.55;
    }
    //if(id==0)
        //printf("in kernel*******\n");

    
}

__global__ void face2full(double *vols, double*faces, int nel, int n, int nxyz, int*iface){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<nel*n){//n = nxyz
        int e = id/n; //+1 in fortran
        //int j = id%n; //+1 in fortran
        int i = iface[id];//[e][j]
        vols[e*nxyz+i] = vols[e*nxyz+i] + faces[id];
    }

}

__global__ void faceu(double *u, double*faces, int toteq, int nel, int n, int nxyz, int*iface){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<toteq*nel*n){
        int ivar = id/(nel*n);
        int e_n = id%(nel*n);
        int e = e_n/n;
        int i = iface[e_n];
        faces[id] = u[e*(toteq*nxyz)+ivar*nxyz+i-1];
    }
}

__global__ void fillq(double *vtrans, double *vx, double *vy, double *vz, double*pr, double*faces, int nel, int n, int nxyz, int*iface, int size){

    //6 faces, each of size nx * nz => n = nx*nz *6
    //vols: e elements each of size nx*ny*nz => nxyz = nx*ny*nz

    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<size){
        int ivar = id/(nel*n);
        int e_n = id%(nel*n);
        int e = e_n/n; //+1 in fortran
        //int j = id%n; //+1 in fortran
        int i = iface[e_n];//[e][j];
	if(ivar==0)
		faces[id] = vtrans[e*nxyz+i-1];
	else if(ivar==1)
		faces[id] = vx[e*nxyz+i-1];
	else if(ivar==2)
                faces[id] = vy[e*nxyz+i-1];
	else if(ivar==3)
                faces[id] = vz[e*nxyz+i-1];
	else if(ivar==4)
                faces[id] = pr[e*nxyz+i-1];

	
        //faces[id]/*[e][j]*/ = vols[ivar*(nxyz*nel)+e*nxyz+i-1]/*[e][i]*/;
        //faces[id] = 2.55;
    }
    //if(id==0)
        //printf("in kernel*******\n");

    
}

extern "C" void faceuwrapper_(int *toteq1, int *n1, int *nxyz1, int*nel1, double *u, double *faces, int *iface){

    int toteq = toteq1[0];
    int n = n1[0];
    int nxyz = nxyz1[0];
    int nel = nel1[0];

    double *d_u, *d_faces;
    int *d_iface;
    bool inCPU = false;
    if(inCPU){
        cudaMalloc(&d_u, nxyz*nel*sizeof(double)*toteq);
        cudaMalloc(&d_iface, n*nel*sizeof(int));
        cudaMalloc(&d_faces, n*nel*sizeof(double)*toteq);
    
        cudaMemcpy( d_u, u, nxyz*nel*sizeof(double)*toteq, cudaMemcpyHostToDevice);
        cudaMemcpy( d_iface, iface, n*nel*sizeof(int), cudaMemcpyHostToDevice);
    }
    else{
        //just assign
        d_u = u;
        d_iface = iface;
        d_faces = faces;
    }
    
    int blockSize, gridSize;
     
    // Number of threads in each thread block
    blockSize = 1024;
              
    // Number of thread blocks in grid
    //gridSize = (int)ceil((float)n*k/blockSize);
    gridSize = (int)ceil((float)n*nel*toteq/blockSize);

    // Execute the kernel
    //printf("block size = %d, grid size = %d\n",blockSize,gridSize);
    faceu<<<gridSize, blockSize>>>(d_u, d_faces, toteq, nel, n, nxyz, d_iface);
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error str 5: %s\n",cudaGetErrorString(code));
    }
    if(inCPU){
        cudaMemcpy( faces, d_faces, n*nel*sizeof(double)*toteq, cudaMemcpyDeviceToHost );
        cudaFree(d_u);
        cudaFree(d_faces);
        cudaFree(d_iface);
    }



}

extern "C" void copyqq_(double*qq, double * faces, int*size){
    cudaMemcpy( faces, qq, size[0]*sizeof(double)*5, cudaMemcpyDeviceToHost );
    cudaFree(qq);
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error in copyqq, str: %s\n",cudaGetErrorString(code));
    }


}

extern "C" void fillqwrapper_(double *vols_vtrans, double *vols_vx, double *vols_vy, double *vols_vz, double *vols_pr, double*faces, int *nel1, int *n1, int *nxyz1, int*iface, bool device_arr, bool pull_result){

    
    int nel = nel1[0];
    int n = n1[0];
    int nxyz = nxyz1[0];
    
    //printf("nel = %d, n = %d, nxyz=%d\n",nel,n,nxyz);
    /*for(int index = 0; index <4; index++){
	    printf("vols_t[%d]=%f,vols_x=%f,vols_y=%f,vols_pr=%f\n",index,vols_vtrans[index],vols_vx[index],vols_vy[index],vols_pr[index]);
	    printf("iface[%d]=%d\n",index,iface[index]);
    }*/

    //double *d_vols_vtrans, *d_vols_vx, d_vols_vy, d_vols_vz, d_vols_pr;
    double *d_vols;
    double *d_vtrans,*d_vx,*d_vy,*d_vz,*d_pr;
    double *d_faces;
    int *d_iface;
    // allocate device vectors memory
    /*cudaMalloc(&d_vols_vtrans, nxyz*nel*sizeof(double));
    cudaMalloc(&d_vols_vx, nxyz*nel*sizeof(double));
    cudaMalloc(&d_vols_vy, nxyz*nel*sizeof(double));
    cudaMalloc(&d_vols_vz, nxyz*nel*sizeof(double));
    cudaMalloc(&d_vols_pr, nxyz*nel*sizeof(double));*/

    bool inCPU = false;
    if(inCPU){
        cudaMalloc(&d_vols, nxyz*nel*sizeof(double)*5);


        cudaMalloc(&d_faces, n*nel*sizeof(double)*5);
        cudaMalloc(&d_iface, n*nel*sizeof(int));

        cudaMemcpy( d_vols, vols_vtrans, nxyz*nel*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy( d_vols+nxyz*nel, vols_vx, nxyz*nel*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy( d_vols+2*nxyz*nel, vols_vy, nxyz*nel*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy( d_vols+3*nxyz*nel, vols_vz, nxyz*nel*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy( d_vols+4*nxyz*nel, vols_pr, nxyz*nel*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy( d_iface, iface, n*nel*sizeof(int), cudaMemcpyHostToDevice);
    }
    else{

        //send vols_vtrans = all vols
        //just assign
        d_vols = vols_vtrans;
	d_vtrans = vols_vtrans;
	d_vx = vols_vx;
	d_vy = vols_vy;
	d_vz = vols_vz;
	d_pr = vols_pr;
        d_faces = faces;
        d_iface = iface;
    }
    
    int blockSize, gridSize;
     
    // Number of threads in each thread block
    blockSize = 1024;
              
    // Number of thread blocks in grid
    //gridSize = (int)ceil((float)n*k/blockSize);
    gridSize = (int)ceil((float)n*nel*5/blockSize);

    // Execute the kernel
    //printf("block size = %d, grid size = %d\n",blockSize,gridSize);
    fillq<<<gridSize, blockSize>>>(d_vtrans,d_vx,d_vy,d_vz,d_pr, d_faces, nel, n, nxyz, d_iface,5*nel*n);
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error str 6: %s\n",cudaGetErrorString(code));
    }
    if(inCPU){
        cudaMemcpy( faces, d_faces, n*nel*sizeof(double)*5, cudaMemcpyDeviceToHost );
        cudaFree(d_faces);
        cudaFree(d_vols);
        cudaFree(d_iface);
    }
    
}


extern "C" void full2facewrapper_(double *vols, double*faces, int *nel1, int *n1, int *ivar1, int *nxyz1, int*iface, bool device_arr, bool pull_result){
	//test printing
        int nel = nel1[0];
        int n = n1[0];
        int nxyz = nxyz1[0];
        int ivar = ivar1[0]-1;
        //printf("nel = %d, n = %d, nxyz=%d, ivar=%d\n",nel,n,nxyz,ivar);
	/*for(int index = 0; index <4; index++){
		printf("vols[%d]=%f\n",index,vols[index]);
		printf("iface[%d]=%d\n",index,iface[index]);
	}*/	


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
    //printf("block size = %d, grid size = %d\n",blockSize,gridSize);
    full2face<<<gridSize, blockSize>>>(d_vols, d_faces, nel, n, nxyz, d_iface);
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error str 7: %s\n",cudaGetErrorString(code));
    }
    cudaMemcpy( faces+ivar*n*nel, d_faces, n*nel*sizeof(double), cudaMemcpyDeviceToHost );	
   	/*for(int index = 0; index <4; index++){
                printf("faces[%d]=%f\n",index,faces[index]);
        }*/
 

	// Release device memory
    cudaFree(d_faces);
    cudaFree(d_vols);
    cudaFree(d_iface);

}

__global__ void rzero (double *arr, int n){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n)
        arr[id] = 0.0;
}

__global__ void surfaceintegral_flux(double * flux, double *area, double *phig, int * index, int toteq, int nelt, int nface, int nxz, int ny){
    int size = toteq * nelt * nface;
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<size){
        int eq = id/(nelt*nface);
        int e_f = id%(nelt*nface);
        int e = e_f/(nface);
        int f = e_f%nface;
        int count = 0;
        for(int i = index[f*6+4]; i < index[f*6+5]; i++){
            for(int j = index[f*6+2]; j <index[f*6+3]; j++){
                for(int k = index[f*6+0]; k < index[f*6+1]; k++){
                    int l = eq*(nelt*nface*nxz)+e*(nface*nxz)+f*(nxz)+count;
                    flux[l] = flux[l] * area[e*(nface*nxz)+f*nxz+count++] * phig[e*ny*nxz+i*nxz+j*ny+k];
                }
            }
        }
    }

}

__global__ void addfull2face(double *vols, double*faces, int nel, int n, int nxyz, int*iface, int size){

    //6 faces, each of size nx * nz => n = nx*nz *6
    //vols: e elements each of size nx*ny*nz => nxyz = nx*ny*nz

    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<size){
        int eq = id/(nel*n);
        int e_n = id%(nel*n);
        int e = e_n/n; //+1 in fortran
        //int j = id%n; //+1 in fortran
        int i = iface[e_n];//[e][j];
        int volIndex = eq*(nel*nxyz)+e*nxyz+i-1;
        vols[volIndex] = vols[volIndex] + faces[id];
    }

    
}

//extern "C" void surfaceintegralwrapper_(int *toteq1, int *nx1, int*ny1, int*nz1, int *nelt1, int *nface1, double* faces, double *area, double *phig, double *vols, int *iface){
extern "C" void surfaceintegralwrapper_(double* faces, double *area, double *phig, double *vols, int *iface, int *toteq1, int *nx1, int*ny1, int*nz1, int *nelt1, int *nface1){

    float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    int nx = nx1[0];
    int ny = ny1[0];
    int nz = nz1[0];
    int nelt = nelt1[0];
    int nface = nface1[0];
    int toteq = toteq1[0];
    //printf("nface = %d, nx = %d, ny = %d, nz = %d, nelt = %d, toteq=%d, faces[0]=%f\n", nface,nx,ny,nz,nelt,toteq,/*faces[nelt*nface*nx*nz-1]*/vols[nelt*nx*ny*nz]);
    int * index = new int[nface*6];
    for(int i =0; i<nface; i++){
        index[i*6+0] = 0;
        index[i*6+1] = nx-1;
        index[i*6+2] = 0;
        index[i*6+3] = ny-1;
        index[i*6+4] = 0;
        index[i*6+5] = nz-1;
    }
    index[0*6+3] = 0;
    index[1*6+0] = nx-1;
    index[2*6+2] = ny-1;
    index[3*6+1] = 0;
    index[4*6+5] = 0;
    index[5*6+4] = nz-1;
    double *d_faces, *d_area, *d_phig, *d_vols;
    int *d_index, *d_iface;
    bool dataInGPU = true;
    cudaError_t code ;
    if(dataInGPU){
        d_faces = faces;
        d_area = area;
        d_phig = phig;
        d_vols = vols;
        d_iface = iface;
    }
    else{
        //memory allocation
        cudaMalloc(&d_faces, toteq*nelt*nx*nz*nface*sizeof(double));
        cudaMalloc(&d_area, nelt*nface*nx*nz*sizeof(double)); //ask about area
        cudaMalloc(&d_phig, nelt*nx*ny*nz*sizeof(double));
        cudaMalloc(&d_vols, toteq*nelt*nx*ny*nz*sizeof(double));
        cudaMalloc(&d_iface, nelt*nface*nx*nz*sizeof(int));
        code = cudaPeekAtLastError();
        if (code != cudaSuccess){
            printf("cuda error in malloc, str: %s\n",cudaGetErrorString(code));
        }


        //data transfer
        
        cudaMemcpy( d_area, area, nelt*nface*nx*nz*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy( d_phig, phig, nelt*nx*ny*nz*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy( d_iface, iface, nelt*nface*nx*nz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( d_faces, faces, toteq*nelt*nx*nz*nface*sizeof(double), cudaMemcpyHostToDevice);


    }

    cudaMalloc(&d_index, nface*6*sizeof(int));
    cudaMemcpy( d_index, index, nface*6*sizeof(int), cudaMemcpyHostToDevice);
    code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error in memcpy, str: %s\n",cudaGetErrorString(code));
    }


    int blockSize, gridSize;
     
    // Number of threads in each thread block
    blockSize = 1024;
              
    // Number of thread blocks in grid
    int ntot = toteq*nelt*nx*ny*nz;
    gridSize = (int)ceil((float)ntot/blockSize);
    rzero<<<gridSize, blockSize>>>(d_vols, ntot);
    code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error in flux, str: %s\n",cudaGetErrorString(code));
    }


    gridSize = (int)ceil((float)toteq*nelt*nface/blockSize);
    surfaceintegral_flux<<<gridSize, blockSize>>>(d_faces, d_area, d_phig, d_index, toteq, nelt, nface, nx*nz, ny);

    code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error in flux, str: %s\n",cudaGetErrorString(code));
    }


    gridSize = (int)ceil((float)toteq*nelt*nx*nz*nface/blockSize);
    addfull2face<<<gridSize, blockSize>>>(d_vols, d_faces, nelt, nx*nz*nface, nx*ny*nz, d_iface,toteq*nelt*nx*nz*nface);
    
    code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error in full2face, str: %s\n",cudaGetErrorString(code));
    }


    if(!dataInGPU){
        cudaMemcpy( vols, d_vols, toteq*nelt*nx*ny*nz*sizeof(double), cudaMemcpyDeviceToHost );
        cudaMemcpy( faces, d_faces, toteq*nelt*nface*nx*nz*sizeof(double), cudaMemcpyDeviceToHost );	
   	// Release device memory
        cudaFree(d_faces);
        cudaFree(d_vols);
        cudaFree(d_iface);
        cudaFree(d_area);
        cudaFree(d_phig);

    }
    cudaFree(d_index);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    //printf("surfcae integral time is %f\n",time*1e-03);
    
}

//mxm multiplication
__global__ void mxm(double *a, int n1, double *b, int n2, double *c, int n3, int nel, int aSize, int bSize, int cSize, int extraEq){

    //calculate c(n1,n3) = a(n1,n2) X b(n2,n3) in c
    //in fortran the original calculation was 
    // c(n3,n1) = b(n3,n2) X a(n2,n1)

    // a,b,cSize are single element size
    //extraEq, in case of a matrix has equation as an index
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<nel*n1*n3){
        int e = id/(n1*n3);
        int rc = id%(n1*n3);
        int i = rc/n3;
        int j = rc%n3;
        int cid = e*cSize + rc;
        int aid = e*aSize + extraEq + i*n2; 
        int bid = e*bSize + j;
        c[cid] = 0;
        for(int k = 0; k<n2; k++)
            c[cid]+=a[aid+k]*b[bid+k*n3];
    }

}
// specmpn routine in fortran
void specmpn(double *d_b, int nb, double *d_a, int na, double * d_ba, double* d_ab, bool if3d, double * d_w, int ldw, int nel, int neq, int eq, bool second_eq){
    //d_a is array(na,na,na)*nel, d_b(nb,nb,nb)*nel, w(ldw)*nel where ldw = na*na*nb+nb*nb*na
    //d_a is array of nel each array(na,na,na)
    int blockSize, gridSize;
     
    // Number of threads in each thread block
    blockSize = 1024;
    cudaStream_t stream;
    cudaStreamCreate( &stream );
    const double alpha = 1;
    const double beta = 0;

    if(if3d){
        int nab = na*nb;
        int nbb = nb*nb;
        //calc w = ba*a in fortran
        //so in c calc wt = at * bat
        //call mxm(ba,nb,a,na,w,na*na)
        //in fortran calc w(nb,na*na) = ba(nb,na) * a(na,na*na)
        //in c w(na*na,nb) = a(na*na,na) * ba(na,nb)
        //neq = 1 if array not indexed by eq and eq = 0
        int aSize = neq*pow(na,3), bSize = pow(nb,3);
        gridSize = (int)ceil((float)na*na*nb*nel/blockSize);
        //mxm<<<gridSize, blockSize>>>(d_a,na*na, d_ba, na, d_w, nb, nel, aSize, 0, ldw, eq*pow(na,3));
        cuda_multi_gemm_unif(stream, 'N', 'N', nb, na, na*na, &alpha, d_ba, nb, 0, d_a, na, aSize, &beta, d_w, nb, ldw, nel, gridSize);
        int k = 0, l = na*na*nb;
        for(int iz=0; iz<na;iz++){
            //calc in fortran wl(nb*nb) = wk(nb*na) * ab(na*nb)
            //in c wl(nb*nb) = ab(nb*na) * wk(na*nb)
            gridSize = (int)ceil((float)nb*nb*nel/blockSize);
            //mxm<<<gridSize, blockSize>>>(d_ab,nb, d_w+k, na, d_w+l, nb, nel, 0, ldw, ldw, 0);
            cuda_multi_gemm_unif(stream, 'N', 'N', nb, na, nb, &alpha, d_w+k, nb, ldw, d_ab, na, 0, &beta, d_w+l, nb, ldw, nel, gridSize);

            k = k + nab;
            l = l + nbb;
        }
        l = na*na*nb;
        //calc in fortran b(nb*nb,nb) = wl(nb*nb,na)* ab(na,nb)
        //in C b(nb,nb*nb) = ab(nb,na) * wl(na,nb*nb)
        gridSize = (int)ceil((float)nb*nb*nb*nel/blockSize);
        //mxm<<<gridSize, blockSize>>>(d_ab,nb, d_w+l, na, d_b, nb*nb, nel, 0, ldw, bSize, 0);
        cuda_multi_gemm_unif(stream, 'N', 'N', nb*nb, na, nb, &alpha, d_w+l, nb*nb, ldw, d_ab, na, 0, &beta, d_b, nb*nb, bSize, nel, gridSize);


    }
    else{
        //calc w(nb*na) = ba(nb,na) * a(na,na) in fortran,
        //in C w(na*nb) = a(na,na) * ba(na,nb)
        gridSize = (int)ceil((float)na*nb*nel/blockSize);
        mxm<<<gridSize, blockSize>>>(d_a,na, d_ba, na, d_w, nb, nel, neq*na*na, 0, ldw, eq*na*na);
        //in fortran, b(nb,nb) = w(nb,na)*ab(na,nb)
        //in C b(nb,nb) = ab(nb,na) * w(na,nb)
        gridSize = (int)ceil((float)nb*nb*nel/blockSize);
        mxm<<<gridSize, blockSize>>>(d_ab,nb, d_w, na, d_b, nb, nel, 0, ldw, nb*nb, 0);


    }
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error str 1: %s\n",cudaGetErrorString(code));
    }
    cudaStreamDestroy(stream);

}

void specmpn_old(double *d_b, int nb, double *d_a, int na, double * d_ba, double* d_ab, bool if3d, double * d_w, int ldw, int nel, int neq, int eq, bool second_eq){
    //d_a is array(na,na,na)*nel, d_b(nb,nb,nb)*nel, w(ldw)*nel where ldw = na*na*nb+nb*nb*na
    //d_a is array of nel each array(na,na,na)
    int blockSize, gridSize;
     
    // Number of threads in each thread block
    blockSize = 1024;

    if(if3d){
        int nab = na*nb;
        int nbb = nb*nb;
        //calc w = ba*a in fortran
        //so in c calc wt = at * bat
        //call mxm(ba,nb,a,na,w,na*na)
        //in fortran calc w(nb,na*na) = ba(nb,na) * a(na,na*na)
        //in c w(na*na,nb) = a(na*na,na) * ba(na,nb)
        //neq = 1 if array not indexed by eq and eq = 0
        int aSize = neq*pow(na,3), bSize = pow(nb,3);
        gridSize = (int)ceil((float)na*na*nb*nel/blockSize);
        mxm<<<gridSize, blockSize>>>(d_a,na*na, d_ba, na, d_w, nb, nel, aSize, 0, ldw, eq*pow(na,3));
        int k = 0, l = na*na*nb;
        for(int iz=0; iz<na;iz++){
            //calc in fortran wl(nb*nb) = wk(nb*na) * ab(na*nb)
            //in c wl(nb*nb) = ab(nb*na) * wk(na*nb)
            gridSize = (int)ceil((float)nb*nb*nel/blockSize);
            mxm<<<gridSize, blockSize>>>(d_ab,nb, d_w+k, na, d_w+l, nb, nel, 0, ldw, ldw, 0);
            k = k + nab;
            l = l + nbb;
        }
        l = na*na*nb;
        //calc in fortran b(nb*nb,nb) = wl(nb*nb,na)* ab(na,nb)
        //in C b(nb,nb*nb) = ab(nb,na) * wl(na,nb*nb)
        gridSize = (int)ceil((float)nb*nb*nb*nel/blockSize);
        mxm<<<gridSize, blockSize>>>(d_ab,nb, d_w+l, na, d_b, nb*nb, nel, 0, ldw, bSize, 0);


    }
    else{
        //calc w(nb*na) = ba(nb,na) * a(na,na) in fortran,
        //in C w(na*nb) = a(na,na) * ba(na,nb)
        gridSize = (int)ceil((float)na*nb*nel/blockSize);
        mxm<<<gridSize, blockSize>>>(d_a,na, d_ba, na, d_w, nb, nel, neq*na*na, 0, ldw, eq*na*na);
        //in fortran, b(nb,nb) = w(nb,na)*ab(na,nb)
        //in C b(nb,nb) = ab(nb,na) * w(na,nb)
        gridSize = (int)ceil((float)nb*nb*nel/blockSize);
        mxm<<<gridSize, blockSize>>>(d_ab,nb, d_w, na, d_b, nb, nel, 0, ldw, nb*nb, 0);


    }
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error str 2: %s\n",cudaGetErrorString(code));
    }

}

__global__ void replicate_3(double *a, int n){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n){
        a[n+id] = a[id];
        a[2*n+id] = a[id];
    }
}

__global__ void nekcol2_conv(double* convh, double *vx, double *vy, double *vz, int n){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n){
        convh[id] = convh[id] * vx[id];
        convh[n+id] = convh[n+id] * vy[id];
        convh[2*n+id] = convh[2*n+id] * vz[id]; 
    }
}

__global__ void merge_replicate_conv(double* convh, double *vx, double *vy, double *vz, int n){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n){
        convh[n+id] = convh[id] * vy[id];
        convh[2*n+id] = convh[id] * vz[id]; 
        convh[id] = convh[id] * vx[id];
    }

    
}

__global__ void nekadd2col2(double *a, double *b, double *c, int n){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n){
        a[id] = a[id] + b[id] * c[id];
    }

}

__global__ void merge_replicate_conv_add2col2(double* convh, double *b, double *c, double *vx, double *vy, double *vz, int n){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n){
        convh[id] = convh[id] + b[id] * c[id];
        convh[n+id] = convh[id] * vy[id];
        convh[2*n+id] = convh[id] * vz[id]; 
        convh[id] = convh[id] * vx[id];
    }

    
}


void evaluate_conv_h(int nel, int neq, int eq, int ndim, int ldw, double *jgl, double *jgt, double * convh, double *u, int nx1, int nxd, int nd, int n1, double *ju1, double*ju2, double*phig, double*pr, double *vxd, double *vyd, double *vzd, double *w, bool if3d){
    //for now totalh = convh so we can pass totalh instead of convh


    //nd = nel * nxd * nyd * nzd
    //n1 = nel * nx1 * ny1 * nz1
    //modify fortran code, convh(nx^3,ndim) -> convh(nx^3,nel,ndim)

    //initially for each element, each equation do
    //do for equation 1
    /*int ldw = 2* pow(2*nxd,ndim);
    double *w;
    cudaMalloc(&w, nel*ldw*sizeof(double));*/

    int nx1_3 = pow(nx1,3);
    if(eq == 0)
        for(int j = 0; j<ndim;j++)
            specmpn(convh+j*nd, nxd, u+(j+1)*nx1_3 ,nx1, jgl, jgt, if3d, w, ldw, nel, neq, j+1, true);
    else{
        specmpn(ju1, nxd, phig, nx1, jgl, jgt, if3d, w, ldw, nel, 1, 0,true);
        specmpn(ju2, nxd, pr, nx1, jgl, jgt, if3d, w, ldw, nel, 1, 0,true);
        if(eq<4){
            specmpn(convh, nxd, u+eq*nx1_3, nx1, jgl, jgt, if3d, w, ldw, nel, neq, eq,true);
            int blockSize, gridSize;
     
            // Number of threads in each thread block
            blockSize = 1024;
            gridSize = (int)ceil((float)nd/blockSize);
            //merge_replicate_conv<<<gridSize, blockSize>>>(convh,vxd,vyd,vzd,nd);
            replicate_3<<<gridSize, blockSize>>>(convh,nd);
            nekcol2_conv<<<gridSize, blockSize>>>(convh,vxd,vyd,vzd,nd);

            nekadd2col2<<<gridSize, blockSize>>>(convh+(eq-1)*nd,ju1,ju2,nd);
        }
        else if(eq==4){
            specmpn(convh, nxd, u+eq*nx1_3, nx1, jgl, jgt, if3d, w, ldw, nel,neq,eq,true);
            int blockSize, gridSize;
     
            // Number of threads in each thread block
            blockSize = 1024;
            gridSize = (int)ceil((float)nd/blockSize);

            //merge_replicate_conv_add2col2<<<gridSize, blockSize>>>(convh,ju1,ju2,vxd,vyd,vzd,nd);

            nekadd2col2<<<gridSize, blockSize>>>(convh,ju1,ju2,nd);
            replicate_3<<<gridSize, blockSize>>>(convh,nd);
            nekcol2_conv<<<gridSize, blockSize>>>(convh,vxd,vyd,vzd,nd);
        }

    }
}

__global__ void nekadd2col2_u(double * u, double *totalh, double *rx, int nel, int n, int ndim, int offset){
    
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n*nel){
        int e = id/n;
        int i = id%n;
        u[id] = 0;
        for(int j = 0; j<ndim; j++)
            u[id] += totalh[j*(nel*n)+id] * rx[e*(3*ndim*n)+(j+offset)*n+i];
    }
}

__global__ void neksub2(double *a, double*b, int n){

    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n)
        a[id]-=b[id];

}

__global__ void nekadd2(double *a, double*b, int n){

    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n)
        a[id]+=b[id];

}

__global__ void nekcol2(double *a, double*b, int n){

    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n)
        a[id]*=b[id];

}

__global__ void nekcol2_ud(double *a, double*b, int nel, int nx1_3, int nxd_3){

    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<nel*nx1_3){
        int e = id/nx1_3;
        int i = id%nx1_3;
        a[e*nxd_3+i]*=b[id];
    }

}


__global__ void nekcopy(double *a, double*b, int n){

    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n)
        a[id]=b[id];

}

extern "C" void nekcopywrapper_(double *a, double *b, int *n){
    int blockSize = 1024, gridSize;
    gridSize = (int)ceil((float)n[0]/blockSize);
    nekcopy<<<gridSize, blockSize>>>(a,b,n[0]); 
}

__global__ void neksubcol3_res1(double *a, double *b, double *c, int nel, int nx1_3){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<nel*nx1_3){
        int i = id%nx1_3;
        a[id] = a[id] - b[i] * c[id];
    }

}

void local_grad3_t(double *u, double *ur, double *us, double *ut, int nxd, double *d, double *dt, double *w, int nel){

    int nxd_2 = nxd * nxd;
    int nxd_3 = nxd_2 * nxd;
    // u(nxd,nxd*nxd) = dt(nxd,nxd) * ur(nxd, nxd*nxd) fortran
    // u(nxd*nxd,nxd) = ur(nxd*nxd, nxd) * dt(nxd,nxd) C
    int blockSize=1024, gridSize;
    cudaStream_t stream;
    cudaStreamCreate( &stream );
    const double alpha = 1;
    const double beta = 0;

    gridSize = (int)ceil((float)nel*nxd_3/blockSize);
    //mxm<<<gridSize, blockSize>>>(ur,nxd_2, dt, nxd, u, nxd, nel, nxd_3, 0, nxd_3, 0);
    cuda_multi_gemm_unif(stream, 'N', 'N', nxd, nxd, nxd_2, &alpha, dt, nxd, 0, ur, nxd, nxd_3, &beta, u, nxd, nxd_3, nel, gridSize);

    for(int k = 0; k<nxd;k++){
        //wk(nxd,nxd) = usk(nxd,nxd)*D(nxd,nxd) fortran
        //wk(nxd,nxd) = D(nxd,nxd)*usk(nxd,nxd) C
        gridSize = (int)ceil((float)nel*nxd_2/blockSize);
        //mxm<<<gridSize, blockSize>>>(d,nxd, us+k*nxd_2, nxd, w+k*nxd_2, nxd, nel, 0, nxd_3, nxd_3, 0);
        cuda_multi_gemm_unif(stream, 'N', 'N', nxd, nxd, nxd, &alpha, us+k*nxd_2, nxd, nxd_3, d, nxd, 0, &beta, w+k*nxd_2, nxd, nxd_3, nel, gridSize);


    }
    gridSize = (int)ceil((float)nel*nxd_3/blockSize);
    nekcopy<<<gridSize, blockSize>>>(u,w, nel*nxd_3);
    //w(nxd*nxd,nxd) = ut(nxd*nxd,nxd) * D(nxd,nxd) fortran
    //w(nxd,nxd*nxd) = D(nxd,nxd) * ut(nxd,nxd*nxd) C
    //mxm<<<gridSize, blockSize>>>(d,nxd, ut, nxd, w, nxd_2, nel, 0, nxd_3, nxd_3, 0);
    cuda_multi_gemm_unif(stream, 'N', 'N', nxd_2, nxd, nxd, &alpha, ut, nxd, nxd_3, d, nxd, 0, &beta, w, nxd_2, nxd_3, nel, gridSize);

    nekadd2<<<gridSize, blockSize>>>(u,w, nel*nxd_3);
    cudaStreamDestroy(stream);



}

void flux_div_integral(double *ur, double *us, double *ut, double *ud, double *tu, double *totalh, double *rx, double *dg, double *dgt, double *jgt, double *jgl, double *res1, double *w, int nel, int eq, int ndim, int nx1, int nxd, int ldw, bool if3d){
    
    //call get_dgl_ptr
    int nd = pow(nxd,3);
    int nx_3 = pow(nx1,3);
    int blockSize = 1024, gridSize;
    gridSize = (int)ceil((float)nel*nd/blockSize);
    nekadd2col2_u<<<gridSize, blockSize>>>(ur, totalh, rx, nel, nd, ndim,0);
    nekadd2col2_u<<<gridSize, blockSize>>>(us, totalh, rx, nel, nd, ndim,ndim);
    if(if3d){
        nekadd2col2_u<<<gridSize, blockSize>>>(ut, totalh, rx, nel, nd, ndim,ndim+ndim);
       local_grad3_t(ud, ur, us, ut, nxd, dg, dgt, w, nel);
    }
    else{
        //call local_grad2
    }
    specmpn(tu,nx1,ud,nxd,jgt,jgl,if3d,w,ldw,nel,1,0,false);
    neksub2<<<gridSize, blockSize>>>(res1+eq*(nel*nx_3),tu,nel*nx_3);

}

void neklocal_grad3(double * ur, double *us, double *ut, double *u, int nx, int nxd, double *d, double *dt, int nel){

    /*double *d_ub;
    cudaMalloc(&d_ub, nel*pow(nx,3)*sizeof(double));
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    const double alpha = 1;
    const double beta = 0;*/

    //cuda_multi_gemm_unif(stream, 'N', 'N', nx, nx, nx, &alpha, nx, nx*nx, d, u, d_ub, &beta, ur, us, ut, nel*nx, 1024);
    //cudaDeviceSynchronize();
    //if(true) return;

    int nx_2 = nx*nx;
    int nx_3 = nx_2*nx;
    int nxd_3 = pow(nxd,3);
    //ur(nx,nx*nx) = D(nx,nx) * u(nx,nx*nx) fortran
    //ur(nx*nx,nx) = u(nx*nx,nx) * D(nx,nx) C
    int blockSize=1024, gridSize;
    gridSize = (int)ceil((float)nel*nx_3/blockSize);
    mxm<<<gridSize, blockSize>>>(u,nx_2, d, nx, ur, nx, nel, nx_3, 0, nxd_3, 0);//ur,us, ut should be indexed by nxd
    for(int k = 0; k<nx; k++){
        //usk(nx,nx) = uk(nx,nx) * dt(nx,nx) fortran
        //usk(nx,nx) = dt(nx,nx) * uk(nx,nx) C
        gridSize = (int)ceil((float)nel*nx_2/blockSize);
    mxm<<<gridSize, blockSize>>>(dt,nx, u+k*nx_2, nx, us+k*nx_2, nx, nel, 0, nx_3, nxd_3, 0);
    }
    //ut(nx_2,nx) = u(nx_2,nx) * dt(nx,nx) fortran
    //ut(nx,nx_2) = dt(nx,nx) * u(nx,nx_2) C
    gridSize = (int)ceil((float)nel*nx_3/blockSize);
    mxm<<<gridSize, blockSize>>>(dt, nx, u, nx, ut, nx_2, nel, 0, nx_3, nxd_3, 0);

}

void nekgradl_rst(double *ur, double *us, double *ut, double *u, double *d, double *dt, int nx, int nxd, int nel, bool if3d){
    if(if3d){
        neklocal_grad3(ur, us, ut, u, nx, nxd, d, dt, nel);
    }
}

__global__ void  calc_ud_3(double *ud, double *rx, double *ur, double *us, double *ut, int j, int nel, int nxd_3){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<nel*nxd_3){
        int e = id/nxd_3;
        int i = id%nxd_3;
        int e_size = e*(9*nxd_3);
        ud[id] = rx[e_size+j*nxd_3+i]*ur[id] + rx[e_size+(j+3)*nxd_3+i]*us[id] + rx[e_size+(j+6)*nxd_3+i]*ut[id];
    }
}

void compute_forcing(double *ud, double *ur, double *us, double *ut, double *phig, double *rx, double *pr, double *convh /*use w*/, double *jacmi, double *bm1, double *res1, double *usrf, double *d, double *dt, int nel, int eq, int nx1, int nxd, int ldim, bool if3d){

    int nxd_2 = nxd*nxd;
    int nx1_2 = nx1 * nx1;
    int nxd_3 = nxd_2*nxd;
    int nx1_3 = nx1_2*nx1;
    int blockSize=1024, gridSize;
    gridSize = (int)ceil((float)nel*nx1_3/blockSize);
    rzero<<<gridSize, blockSize>>>(ud,nel*nx1_3);
    if(eq!=0&&eq!=4){
        int j=0;
        if(eq==2)
            j=1;
        else if(eq==3){
            j=1;
            if(ldim==3)
                j=2;
        }
        nekgradl_rst(ur,us,ut,phig, d, dt,nx1, nxd, nel, if3d);
        if(if3d){
            gridSize = (int)ceil((float)nel*nxd_3/blockSize);
            calc_ud_3<<<gridSize, blockSize>>>(ud,rx,ur,us,ut,j,nel,nxd_3);
        }
        else{
            //calc_ud_2
        }
        if(eq!=3 || ldim!=2){
            gridSize = (int)ceil((float)nel*nx1_3/blockSize);
            nekcol2_ud<<<gridSize, blockSize>>>(ud,pr,nel,nx1_3,nxd_3);
            nekcopy<<<gridSize, blockSize>>>(convh,ud,nel*nx1_3);
            nekcol2<<<gridSize, blockSize>>>(convh,jacmi,nel*nx1_3);
            nekcol2<<<gridSize, blockSize>>>(convh,bm1,nel*nx1_3);
            neksub2<<<gridSize, blockSize>>>(res1+eq*(nel*nx1_3),convh,nel*nx1_3);
            neksubcol3_res1<<<gridSize, blockSize>>>(res1+eq*(nel*nx1_3),usrf+eq*nx1_3, bm1,nel,nx1_3);



        }
    }
    else if (eq==4)
        neksubcol3_res1<<<gridSize, blockSize>>>(res1+eq*(nel*nx1_3),usrf+eq*nx1_3, bm1,nel,nx1_3);

}


// this function is doing assemble_h, flux_div_integral, compute_forcing
extern "C" void computestagewrapper_(double *jgl, double *jgt, double *totalh, double *u, double *ju1, double *ju2, double *phig, double*pr, double *vxd, double *vyd, double *vzd, double *ut, double *ud, double *tu, double *rx, double *dg, double *dgt, double *res1, double *w, double *jacmi, double *bm1, double *usrf, double *d, double *dt, int *nel1, int *neq1, int *ndim1, int *ldw1, int *nx11, int *nxd1/*, bool if3d*/){

    float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    bool if3d = true;
    int nel = nel1[0];
    int neq = neq1[0];
    int ndim = ndim1[0];
    int ldw = ldw1[0];
    int nx1 = nx11[0];
    int nxd = nxd1[0];
    //printf("nel=%d,neq=%d,ndim=%d,nx1=%d,nxd=%d,u[0]=%f\n",nel,neq,ndim,nx1,nxd,u[0]);
    //use d_ju1 for d_ur
    double *d_jgl, *d_jgt, *d_totalh, *d_u, *d_ju1, *d_ju2, *d_phig, *d_pr, *d_vxd,*d_vyd, *d_vzd, *d_ut, *d_ud, *d_tu, *d_rx, *d_dg, *d_dgt, *d_res1, *d_w, *d_jacmi, *d_bm1, *d_usrf, *d_d, *d_dt ;
    bool inCPU = false;
    int nxd_3 = pow(nxd,3), nx1_3 = pow(nx1,3);    
    if(inCPU){
        //copy data to gpu
        cudaMalloc(&d_jgl, nxd_3*sizeof(double));
        cudaMemcpy(d_jgl, jgl, nxd_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_jgt, nxd_3*sizeof(double));
        cudaMemcpy(d_jgt, jgt, nxd_3*sizeof(double), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_totalh, ndim*nel*nxd_3*sizeof(double));
        cudaMalloc(&d_u, nel*neq*nx1_3*sizeof(double));
        cudaMemcpy(d_u, u, nel*neq*nx1_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ju1, nel*nxd_3*sizeof(double));
        cudaMalloc(&d_ju2, nel*nxd_3*sizeof(double));

        cudaMalloc(&d_phig, nel*nx1_3*sizeof(double));
        cudaMemcpy(d_phig, phig, nel*nx1_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_pr, nel*nx1_3*sizeof(double));
        cudaMemcpy(d_pr, pr, nel*nx1_3*sizeof(double), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_vxd, nel*nxd_3*sizeof(double));
        cudaMemcpy(d_vxd, vxd, nel*nxd_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_vyd, nel*nxd_3*sizeof(double));
        cudaMemcpy(d_vyd, vyd, nel*nxd_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_vzd, nel*nxd_3*sizeof(double));
        cudaMemcpy(d_vzd, vzd, nel*nxd_3*sizeof(double), cudaMemcpyHostToDevice);

        cudaMalloc(&d_ut, nel*nxd_3*sizeof(double));
        cudaMalloc(&d_ud, nel*nxd_3*sizeof(double));
        cudaMalloc(&d_tu, nel*nxd_3*sizeof(double));
        
        cudaMalloc(&d_rx, nel*9*nxd_3*sizeof(double));
        cudaMemcpy(d_rx, rx, nel*9*nxd_3*sizeof(double), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_dg, nxd_3*sizeof(double));
        cudaMemcpy(d_dg, dg, nxd_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_dgt, nxd_3*sizeof(double));
        cudaMemcpy(d_dgt, dgt, nxd_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_d, nxd_3*sizeof(double));
        cudaMemcpy(d_d, d, nxd_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_dt, nxd_3*sizeof(double));
        cudaMemcpy(d_dt, dt, nxd_3*sizeof(double), cudaMemcpyHostToDevice);

        

        cudaMalloc(&d_res1, nel*neq*nx1_3*sizeof(double));
        cudaMemcpy(d_res1, res1, nel*neq*nx1_3*sizeof(double), cudaMemcpyHostToDevice);

        cudaMalloc(&d_w, nel*ldw*sizeof(double));
        //cudaMemcpy(d_w, w, ldw*sizeof(double), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_jacmi, nel*nx1_3*sizeof(double));
        cudaMemcpy(d_jacmi, jacmi, nel*nx1_3*sizeof(double), cudaMemcpyHostToDevice);

        cudaMalloc(&d_bm1, nel*nx1_3*sizeof(double));
        cudaMemcpy(d_bm1, bm1, nel*nx1_3*sizeof(double), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_usrf, neq*nx1_3*sizeof(double));
        cudaMemcpy(d_usrf, usrf, neq*nx1_3*sizeof(double), cudaMemcpyHostToDevice);
        

    }
    else{
        //just assign pointers
        
        d_jgl = jgl;
        d_jgt = jgt;
        d_totalh = totalh;
        d_u = u;
        d_ju1 = ju1;
        d_ju2 = ju2;
        d_phig = phig;
        d_pr = pr;
        d_vxd = vxd;
        d_vyd = vyd;
        d_vzd = vzd;
        d_ut = ut;
        d_ud = ud;
        d_tu = tu;
        d_rx = rx;
        d_dg = dg;
        d_dgt = dgt;
        d_res1 = res1;
        d_w = w;
        d_jacmi = jacmi;
        d_bm1 = bm1;
        d_usrf = usrf;
        d_d = d;
        d_dt = dt;
    }
    //printf("finished memory allocation in compute\n eq=%d, if3d=%d\n",neq,if3d);
    for(int eq = 0; eq<neq; eq++){
        //printf("loop # %d\n",eq);
        evaluate_conv_h(nel, neq, eq, ndim, ldw, d_jgl, d_jgt, d_totalh /*convh*/, d_u, nx1, nxd, /*nd*/ pow(nxd,3)*nel, /*n1*/ pow(nx1,3)*nel, d_ju1, d_ju2, d_phig, d_pr, d_vxd, d_vyd, d_vzd, d_w, if3d);
        
        flux_div_integral(d_ju1/*d_ur*/, d_ju2/*d_us*/, d_ut, d_ud, d_tu, d_totalh, d_rx, d_dg, d_dgt, d_jgt, d_jgl, d_res1, d_w, nel, eq, ndim, nx1, nxd, ldw, if3d);

        compute_forcing(d_ud, d_ju1/*d_ur*/, d_ju2/*d_us*/, d_ut, d_phig, d_rx, d_pr, d_w /*convh*/, d_jacmi, d_bm1, d_res1, d_usrf, d_d, d_dt, nel, eq, nx1, nxd, ndim/*ldim*/, if3d);


    }
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error str 3: %s\n",cudaGetErrorString(code));
    }

    if(inCPU){
        cudaMemcpy(res1, d_res1, nel*neq*nx1_3*sizeof(double), cudaMemcpyDeviceToHost);
        //cuda free all d_*
        //double *d_jgl, *d_jgt, *d_totalh, *d_u, *d_ju1, *d_ju2, *d_phig, *d_pr, *d_vxd,*d_vyd, *d_vzd, *d_ut, *d_ud, *d_tu, *d_rx, *d_dg, *d_dgt, *d_res1, *d_w, *d_jacmi, *d_bm1, *d_usrf, *d_d, *d_dt ;

        cudaFree(d_jgl);
        cudaFree(d_jgt);
        cudaFree(d_totalh);
        cudaFree(d_u);
        cudaFree(d_ju1);
        cudaFree(d_ju2);
        cudaFree(d_phig);
        cudaFree(d_pr);
        cudaFree(d_vxd);
        cudaFree(d_vyd);
        cudaFree(d_vzd);
        cudaFree(d_ut);
        cudaFree(d_ud);
        cudaFree(d_tu);
        cudaFree(d_rx);

        cudaFree(d_dg);
        cudaFree(d_dgt);
        cudaFree(d_d);
        cudaFree(d_dt);        
        cudaFree(d_res1);
        cudaFree(d_w); 
        cudaFree(d_jacmi);
        cudaFree(d_bm1); 
        cudaFree(d_usrf);


    }
    else{
    } 
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    //printf("compute stage time is %f\n",time*1e-03);
}

__global__ void calculate_u(double *u, double *bm1, double *tcoef, double *res3, double *res1, int nelt, int nxyz1, int toteq){

    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<nelt*toteq*nxyz1){
        int e = id/(toteq*nxyz1);
        int r = id%(toteq*nxyz1);
        int eq = r/nxyz1;
        int i = r%nxyz1;
        u[id] = bm1[e*nxyz1+i]*tcoef[0]*res3[id]+bm1[e*nxyz1+i]*tcoef[1]*u[id]-tcoef[2]*res1[eq*(nelt*nxyz1)+e*nxyz1+i];
        u[id] = u[id]/bm1[e*nxyz1+i];
    }
 
}


extern "C" void calculateuwrapper_(double *u, double *bm1, double *tcoef, double *res3, double *res1, int *stage1, int *nelt1, int *nxyz11, int *toteq1){

    int stage = stage1[0]-1;
    int nelt = nelt1[0];
    int nxyz1 = nxyz11[0];
    int toteq = toteq1[0];
    bool inCPU = false;
    if(inCPU){
        double *d_u, *d_bm1, *d_tcoef, *d_res3, *d_res1;
        cudaMalloc(&d_u, nelt*toteq*nxyz1*sizeof(double));
        cudaMemcpy(d_u, u, nelt*toteq*nxyz1*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_bm1, nelt*nxyz1*sizeof(double));
        cudaMemcpy(d_bm1, bm1, nelt*nxyz1*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_tcoef, 9*sizeof(double));
        cudaMemcpy(d_tcoef, tcoef, 9*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_res3, nelt*toteq*nxyz1*sizeof(double));
        cudaMemcpy(d_res3, res3, nelt*toteq*nxyz1*sizeof(double), cudaMemcpyHostToDevice);

        cudaMalloc(&d_res1, nelt*toteq*nxyz1*sizeof(double));
        cudaMemcpy(d_res1, res1, nelt*toteq*nxyz1*sizeof(double), cudaMemcpyHostToDevice);

        int blockSize = 1024, gridSize;
        gridSize = (int)ceil((float)nelt*toteq*nxyz1/blockSize);
        calculate_u<<<gridSize, blockSize>>>(d_u,d_bm1,d_tcoef+stage*3,d_res3,d_res1,nelt,nxyz1,toteq);
        cudaMemcpy(u, d_u, nelt*toteq*nxyz1*sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_u);
        cudaFree(d_bm1);
        cudaFree(d_tcoef);
        cudaFree(d_res3);
        cudaFree(d_res1);

    }
    else{
        int blockSize = 1024, gridSize;
        gridSize = (int)ceil((float)nelt*toteq*nxyz1/blockSize);
        calculate_u<<<gridSize, blockSize>>>(u,bm1,tcoef+stage*3,res3,res1,nelt,nxyz1,toteq);

    }
    
}

__global__ void nekinvcol3_vu(double *vx, double *vy, double *vz, double *u, int nel, int nxyz1, int neq, int irg, int irpu, int irpv, int irpw){    
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<nel*nxyz1){
        int e = id/nxyz1;
        int i = id%nxyz1;
        int e_offset = neq*nxyz1;
        double c = u[e*e_offset+irg*nxyz1+i];
        vx[id] = u[e*e_offset+irpu*nxyz1+i]/c;
        vy[id] = u[e*e_offset+irpv*nxyz1+i]/c;
        vz[id] = u[e*e_offset+irpw*nxyz1+i]/c;
        vx[id] = 0.0;
        vy[id] = 1.0;
        vz[id] = 0.0;

    }
}

extern "C" void computeprimitivevarswrapper_(double *vx, double *vy, double *vz, double *vxd, double *vyd, double *vzd, double *u, double *jgl, double *jgt, double *w, int *nxd1, int *nx11, int *nel1, int *toteq1, int *irpu1, int *irpv1, int *irpw1, int *irg1, int *ldw1, int *p){
    //called only once and values used for all equations
    float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    int nxd = nxd1[0];
    int nx1 = nx11[0];
    int nel = nel1[0];
    int toteq = toteq1[0];
    int irpu = irpu1[0]-1;
    int irpv = irpv1[0]-1;
    int irpw = irpw1[0]-1;
    int irg = irg1[0]-1;
    int nx1_3 = pow(nx1,3);
    int ldw = ldw1[0];

    double *d_vx, *d_vy, *d_vz, *d_vxd, *d_vyd, *d_vzd, *d_u, *d_jgl, *d_jgt, *d_w;
    bool inCPU = false; 
    if(p[0]==1)
      inCPU = true;
    if(inCPU){
        //allocate gpu memory and transfer data to GPU
        int tot_b = nel * nx1_3 * sizeof(double);
        int totd_b = nel * pow(nxd,3) * sizeof(double);
        ldw = 2*pow(nxd,3);
        int ldw_b = nel * ldw * sizeof(double);
        cudaMalloc(&d_vx, tot_b);
        cudaMalloc(&d_vy, tot_b);
        cudaMalloc(&d_vz, tot_b);

        cudaMalloc(&d_vxd, totd_b);
        cudaMalloc(&d_vyd, totd_b);
        cudaMalloc(&d_vzd, totd_b);

        cudaMalloc(&d_w, ldw_b);

        cudaMalloc(&d_u, toteq*tot_b);
        int nxd_3_b = pow(nxd,3) * sizeof(double);
        cudaMalloc(&d_jgl, nxd_3_b);
        cudaMalloc(&d_jgt, nxd_3_b);
        cudaMemcpy(d_u, u, toteq*tot_b, cudaMemcpyHostToDevice);
        cudaMemcpy(d_jgl, jgl, nxd_3_b, cudaMemcpyHostToDevice);
        cudaMemcpy(d_jgt, jgt, nxd_3_b, cudaMemcpyHostToDevice);



    }
    else{
        //just assign data
        d_w = w;
        d_vx = vx;
        d_vy = vy;
        d_vz = vz;

        d_vxd = vxd;
        d_vyd = vyd;
        d_vzd = vzd;
        d_u = u;
        d_jgl = jgl;
        d_jgt = jgt;
    }

    int blockSize = 1024, gridSize;
    gridSize = (int)ceil((float)nel*nx1_3/blockSize);
    nekinvcol3_vu<<<gridSize, blockSize>>>(d_vx, d_vy, d_vz, d_u, nel, nx1_3, toteq, irg, irpu, irpv, irpw);

    specmpn(d_vxd, nxd, d_vx, nx1, d_jgl, d_jgt, true, d_w, ldw, nel, 1, 0,true);
    specmpn(d_vyd, nxd, d_vy, nx1, d_jgl, d_jgt, true, d_w, ldw, nel, 1, 0,true);
    specmpn(d_vzd, nxd, d_vz, nx1, d_jgl, d_jgt, true, d_w, ldw, nel, 1, 0,true);

    if(inCPU){
        int tot_b = nel * nx1_3 * sizeof(double);
        int totd_b = nel * pow(nxd,3) * sizeof(double);

        cudaMemcpy(vx, d_vx, tot_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(vy, d_vy, tot_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(vz, d_vz, tot_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(vxd, d_vxd, totd_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(vyd, d_vyd, totd_b, cudaMemcpyDeviceToHost);
        cudaMemcpy(vzd, d_vzd, totd_b, cudaMemcpyDeviceToHost);
 

        cudaFree(d_vx);
        cudaFree(d_vy);
        cudaFree(d_vz);
        cudaFree(d_vxd);
        cudaFree(d_vyd);
        cudaFree(d_vzd);

        cudaFree(d_w);

        cudaFree(d_u);
        cudaFree(d_jgl);
        cudaFree(d_jgt);



    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    //printf("compute primitive time is %f\n",time*1e-03);


}

//mxm multiplication, faces
__global__ void mxm_faces(double *a, int n1, double *b, int n2, double *c, int n3, int nel, int nfaces, int aSize, int bSize, int cSize){

    //calculate c(n1,n3) = a(n1,n2) X b(n2,n3) in c
    //in fortran the original calculation was 
    // c(n3,n1) = b(n3,n2) X a(n2,n1)

    // a,b,cSize are single element size
    //extraEq, in case of a matrix has equation as an index
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int n13 = n1*n3;
    if(id<nel*nfaces*n13){
        int e = id/(nfaces*n13);
        int rc = id%(nfaces*n13);
        int f = rc/n13;
        int rc2 = rc%n13;
        int i = rc2/n3;
        int j = rc2%n3;
        int cid = e*nfaces*cSize+f*cSize+rc2;
        int aid = e*nfaces*aSize+f*aSize + i*n2; 
        int bid = e*nfaces*bSize+f*bSize + j;
        c[cid] = 0;
        for(int k = 0; k<n2; k++)
            c[cid]+=a[aid+k]*b[bid+k*n3];
    }

}

void map_faced(double *jgl, double *jgt, double *ju, double *u, double *w, int nx1, int nxd, int fdim, int nelt, int nfaces, int idir){

    cudaStream_t stream;
    cudaStreamCreate( &stream );
    const double alpha = 1;
    const double beta = 0;
    int nx1_2 = pow(nx1,2);
    int nxd_2 = pow(nxd,2);
    int batchSize = nelt*nfaces;


    if(idir==0){
        int blockSize = 1024, gridSize;
        //calc w(nxd,nx1) = jgl(nxd*nx1) * u(nx1,nx1) in fortran
        //calc w(nx1,nxd) = u(nx1,nx1) * jgl(nx1,nxd) in C
        gridSize = (int)ceil((float)nelt*nfaces*nx1*nxd/blockSize);
        cuda_multi_gemm_unif(stream, 'N', 'N', nxd, nx1, nx1, &alpha, jgl, nxd, 0, u, nx1, nx1_2, &beta, w, nxd, nx1*nxd, batchSize, gridSize);
        //mxm_faces<<<gridSize, blockSize>>>(u, nx1, jgl, nx1, w, nxd, nelt, nfaces, nx1*nx1, 0, nx1*nxd);
        
        //calc ju(nxd,nxd) = w(nxd,nx1) * jgt(nx1,nxd) in fortran
        //calc ju(nxd,nxd) = jgt(nxd,nx1) * w(nx1,nxd)
        gridSize = (int)ceil((float)nelt*nfaces*nxd*nxd/blockSize);
        cuda_multi_gemm_unif(stream, 'N', 'N', nxd, nx1, nxd, &alpha, w, nxd, nx1*nxd, jgt, nx1, 0, &beta, ju, nxd, nxd_2, batchSize, gridSize);
        //mxm_faces<<<gridSize, blockSize>>>(jgt, nxd, w, nx1, ju, nxd, nelt, nfaces, 0, nx1*nxd, nxd*nxd);
    }
    else{
        int blockSize = 1024, gridSize;
        //calc w(nx1,nxd) = jgt(nx1,nxd) * u(nxd,nxd) in fortran
        //calc w(nxd,nx1) = u(nxd,nxd) * jgt(nxd,nx1) in C
        gridSize = (int)ceil((float)nelt*nfaces*nx1*nxd/blockSize);
        cuda_multi_gemm_unif(stream, 'N', 'N', nx1, nxd, nxd, &alpha, jgt, nx1, 0, u, nxd, nxd_2, &beta, w, nx1, nx1*nxd, batchSize, gridSize);
        //mxm_faces<<<gridSize, blockSize>>>(u, nxd, jgt, nxd, w, nx1, nelt, nfaces, nxd*nxd, 0, nx1*nxd);

        //calc ju(nx1,nx1) = w(nx1,nxd) * jgl(nxd,nx1) in fortran
        //calc ju(nx1,nx1) = jgl(nx1,nxd) * w(nxd,nx1) in C
        gridSize = (int)ceil((float)nelt*nfaces*nx1*nx1/blockSize);
        cuda_multi_gemm_unif(stream, 'N', 'N', nx1, nxd, nx1, &alpha, w, nx1, nx1*nxd, jgl, nxd, 0, &beta, ju, nx1, nx1_2, batchSize, gridSize);
        //mxm_faces<<<gridSize, blockSize>>>(jgl, nx1, w, nxd, ju, nx1, nelt, nfaces, 0, nx1*nxd, nx1*nx1);

    }
    cudaStreamDestroy(stream);
}

__global__ void invcol3_flux(double *a, double *b, double *c, int n, int total){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<total){
        a[id] = b[id] / c[id%n];
    }
}

__global__ void nekcol2_flux(double *a, double*b, int n, int total){

    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<total)
        a[id]*=b[id%n];

}

__global__ void invcol2(double *a, double*b, int n){

    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n)
        a[id]= a[id]/b[id];

}

__global__ void Ausm_flux(int neq, int ntotd, double *nx, double *ny, double *nz, double *nm, double *fs, double *rl, double *ul, double *vl, double *wl, double *pl, double *al, double *tl, double *rr, double *ur, double *vr, double *wr, double *pr, double *ar, double *tr, double *flx, double *cpl, double *cpr){

    int i = blockIdx.x*blockDim.x+threadIdx.x;
    //ntotd = nel * nfaces * nxzd
    if(i<ntotd){
        fs[i] = 0;// it is 0 in cmtbone but can be changed
        double af,mf,mfa,mfm,mfp,ml,mla,mlp,mr,mra,mrm,pf,ql,qr,wtl,wtr,Hl,Hr;
        Hl = cpl[i]*tl[i] + 0.5*(ul[i]*ul[i]+vl[i]*vl[i]+wl[i]*wl[i]);
        Hr = cpr[i]*tr[i] + 0.5*(ur[i]*ur[i]+vr[i]*vr[i]+wr[i]*wr[i]);

        ql = ul[i]*nx[i] + vl[i]*ny[i] + wl[i]*nz[i] - fs[i];
        qr = ur[i]*nx[i] + vr[i]*ny[i] + wr[i]*nz[i] - fs[i];
        
        af = 0.5*(al[i] + ar[i]);
        ml = ql/af;
        mla = abs(ml);

        mr = qr/af;
        mra = abs(mr);

        if(mla <= 1.0){
            mlp = 0.25*pow((ml+1.0),2) + 0.125*pow((ml*ml-1.0),2);
            wtl = 0.25*pow(ml+1.0,2)*(2.0-ml) + 0.1875*ml*pow(ml*ml-1.0,2);
        }
        else{
            mlp = 0.5*(ml+mla);
            wtl = 0.5*(1.0+ml/mla);
        }
        if(mra <= 1.0){
            mrm = -0.25*pow((mr-1.0),2) - 0.125*pow((mr*mr-1.0),2);
            wtr = 0.25*pow(mr-1.0,2)*(2.0+mr) - 0.1875*mr*pow(mr*mr-1.0,2);
        }
        else{
            mrm = 0.5*(mr-mra);
            wtr = 0.5*(1.0-mr/mra);
        }
                
        mf = mlp + mrm;
        mfa = abs(mf);
        mfp = 0.5*(mf+mfa);
        mfm = 0.5*(mf-mfa);

        pf = wtl*pl[i] + wtr*pr[i];

        //compute fluxes
        flx[i] = (af*(mfp*rl[i] + mfm*rr[i])) * nm[i];
        flx[1*ntotd+i] = (af*(mfp*rl[i]*ul[i] + mfm*rr[i]*ur[i])+pf*nx[i]) * nm[i];
        flx[2*ntotd+i] = (af*(mfp*rl[i]*vl[i] + mfm*rr[i]*vr[i])+pf*ny[i]) * nm[i];
        flx[3*ntotd+i] = (af*(mfp*rl[i]*wl[i] + mfm*rr[i]*wr[i])+pf*nz[i]) * nm[i];
        flx[4*ntotd+i] = (af*(mfp*rl[i]*Hl + mfm*rr[i]*Hr)+pf*fs[i]) * nm[i];


    }
}

void InviscidFlux(double *qminus, double *qplus, double *flux, double *unx, double *uny, double *unz, double *area, double *wghtc, double *wghtf, double *cbc, double *jgl, double *jgt, double *nx, double *ny, double *nz, double *rl, double *ul, double *vl, double *wl, double *pl, double *tl, double *al, double *cpl, double *rr, double *ur, double *vr, double *wr, double *pr, double *tr, double *ar, double *cpr, double *fs, double *jaco_f, double *flx, double *jaco_c,int neq, int nstate, int nflux, int nxd, int nx1, int nel, int ndim, int irho, int iux, int iuy, int iuz, int ipr, int ithm, int isnd, int icpf, int iph){

    //nx extended to be nx(nel,nfaces,#points_in_face)
    //irho should be irho1[0]-1, others also
    //printf("in invFlux**\n");
    int fdim = ndim-1;
    int nfaces = 2*ndim;
    int nx1_2 = nx1*nx1;
    int nxd_2 = nxd*nxd;
    double *w;
    cudaMalloc(&w,nel*nfaces*pow(nxd,2)*sizeof(double));


    //add neksub2 which is last step of face_state_commo
    int blockSize1 = 1024, gridSize1;
    gridSize1 = (int)ceil((float)nstate*nel*nfaces*nx1_2/blockSize1);

    neksub2<<<gridSize1, blockSize1>>>(qplus,qminus,nstate*nel*nfaces*nx1_2);

    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error Inv, comp-1: %s\n",cudaGetErrorString(code));
    }


    int totpts = nel * nfaces *  nx1_2;
    map_faced(jgl, jgt, nx, unx, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, ny, uny, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, nz, unz, w, nx1, nxd, fdim, nel, nfaces, 0);
    code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error Inv, comp-2: %s\n",cudaGetErrorString(code));
    }


    //printf("irho=%d,iux=%d,iuy=%d,iuz=%d,ipr=%d,ithm=%d,isnd=%d,icpf=%d\n",irho,iux,iuy,iuz,ipr,ithm,isnd,icpf);
    map_faced(jgl, jgt, rl, qminus+irho*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, ul, qminus+iux*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, vl, qminus+iuy*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, wl, qminus+iuz*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, pl, qminus+ipr*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, tl, qminus+ithm*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, al, qminus+isnd*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, cpl, qminus+icpf*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error Inv, comp-3: %s\n",cudaGetErrorString(code));
    }

    
    map_faced(jgl, jgt, rr, qplus+irho*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, ur, qplus+iux*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, vr, qplus+iuy*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, wr, qplus+iuz*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, pr, qplus+ipr*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, tr, qplus+ithm*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, ar, qplus+isnd*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    map_faced(jgl, jgt, cpr, qplus+icpf*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error Inv, comp-4: %s\n",cudaGetErrorString(code));
    }


    int blockSize = 1024, gridSize;
    gridSize = (int)ceil((float)totpts/blockSize);
    invcol3_flux<<<gridSize, blockSize>>>(jaco_c,area,wghtc,nx1_2,totpts);
    map_faced(jgl, jgt, jaco_f, jaco_c, w, nx1, nxd, fdim, nel, nfaces, 0);
    
    int totpts_d = nel * nfaces * nxd_2;
    gridSize = (int)ceil((float)totpts_d/blockSize);
    nekcol2_flux<<<gridSize, blockSize>>>(jaco_f,wghtf,nxd_2,totpts_d);

    //Ausm
    //gridSize = (int)ceil((float)nel*nfaces*nxd_2/blockSize);
    invcol2<<<gridSize, blockSize>>>(cpl,rl,totpts_d);
    invcol2<<<gridSize, blockSize>>>(cpr,rr,totpts_d);

    //gridSize = (int)ceil((float)nel*nfaces*nxd_2/blockSize);
    Ausm_flux<<<gridSize, blockSize>>>(neq, totpts_d, nx, ny, nz, jaco_f, fs, rl, ul, vl, wl, pl, al, tl, rr, ur, vr, wr, pr, ar, tr, flx, cpl, cpr);
    code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error Inv, comp-5: %s\n",cudaGetErrorString(code));
    }


    map_faced(jgl, jgt, pl, qminus+iph*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
    for(int j=0; j<neq;j++){
        nekcol2<<<gridSize, blockSize>>>(flx+j*totpts_d,pl,totpts_d);
        map_faced(jgl, jgt, flux+j*totpts, flx+j*totpts_d, w, nx1, nxd, fdim, nel, nfaces, 1);
    }
    code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error Inv, comp-6: %s\n",cudaGetErrorString(code));
    }


    cudaFree(w);
    code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error Inv, comp-7: %s\n",cudaGetErrorString(code));
    }

}

extern "C" void inviscidfluxwrapper_(double *qminus, double *qplus, double *flux, double *unx, double *uny, double *unz, double *area, double *wghtc, double *wghtf, double *cbc, double *jgl, double *jgt, double *nx, double *ny, double *nz, double *rl, double *ul, double *vl, double *wl, double *pl, double *tl, double *al, double *cpl, double *rr, double *ur, double *vr, double *wr, double *pr, double *tr, double *ar, double *cpr, double *fs, double *jaco_f, double *flx, double *jaco_c, int* neq, int* nstate, int* nflux, int *nxd, int *nx1, int *nel, int *ndim, int *irho, int *iux, int *iuy, int *iuz, int *ipr, int *ithm, int *isnd, int *icpf, int *iph){
    bool inCPU = false;
    if(inCPU){
        //input and output
        double *d_qminus, *d_qplus, *d_flux, *d_unx, *d_uny, *d_unz, *d_area, *d_wghtc, *d_wghtf, *d_cbc, *d_jgl, *d_jgt;
        //temp arrays
        double *d_nx, *d_ny, *d_nz, *d_rl, *d_ul, *d_vl, *d_wl, *d_pl, *d_tl, *d_al, *d_cpl, *d_rr, *d_ur, *d_vr, *d_wr, *d_pr, *d_tr, *d_ar, *d_cpr, *d_fs, *d_jaco_f, *d_flx, *d_jaco_c;
        int nfaces=ndim[0]*2;
        int ntot = nel[0] * nfaces * pow(nx1[0],2);
        int ntotd = nel[0] * nfaces * pow(nxd[0],2);
        cudaMalloc(&d_qminus, nstate[0]*ntot*sizeof(double));
        cudaMemcpy(d_qminus, qminus, nstate[0]*ntot*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_qplus, nstate[0]*ntot*sizeof(double));
        cudaMemcpy(d_qplus, qplus, nstate[0]*ntot*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_flux, neq[0]*ntot*sizeof(double));
        
        cudaMalloc(&d_unx, ntot*sizeof(double));
        cudaMemcpy(d_unx, unx, ntot*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_uny, ntot*sizeof(double));
        cudaMemcpy(d_uny, uny, ntot*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_unz, ntot*sizeof(double));
        cudaMemcpy(d_unz, unz, ntot*sizeof(double), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_area, ntot*sizeof(double));
        cudaMemcpy(d_area, area, ntot*sizeof(double), cudaMemcpyHostToDevice);

        cudaMalloc(&d_wghtc, pow(nx1[0],2)*sizeof(double));
        cudaMemcpy(d_wghtc, wghtc, pow(nx1[0],2)*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_wghtf, pow(nxd[0],2)*sizeof(double));
        cudaMemcpy(d_wghtf, wghtf, pow(nxd[0],2)*sizeof(double), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_cbc, pow(nxd[0],2)*sizeof(double));//correct
        //cudaMemcpy(d_wghtf, wghtf, pow(nxd[0],2)*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_jgl, pow(nxd[0],3)*sizeof(double));
        cudaMemcpy(d_jgl, jgl, pow(nxd[0],3)*sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_jgt, pow(nxd[0],3)*sizeof(double));
        cudaMemcpy(d_jgt, jgt, pow(nxd[0],3)*sizeof(double), cudaMemcpyHostToDevice);


	 double*d_all;
	 cudaMalloc(&d_all, 26*ntotd*sizeof(double));
	 d_nx = d_all;
	d_ny = d_nx+ntotd;
	d_nz = d_ny+ntotd;
        d_rl = d_nz+ntotd;
	d_ul = d_rl+ntotd;
	d_wl = d_ul+ntotd;
	d_vl = d_wl+ntotd;
	d_pl = d_vl+ntotd;
	d_tl = d_pl+ntotd;
	d_al = d_tl+ntotd;
	d_cpl = d_al+ntotd;
	d_rr = d_cpl+ntotd;
	d_ur = d_rr+ntotd;
	d_wr = d_ur+ntotd;
	d_vr = d_wr+ntotd;
	d_pr = d_vr+ntotd;
	d_tr = d_pr+ntotd;
	d_ar = d_tr+ntotd;
	d_cpr = d_ar+ntotd;
	d_jaco_f = d_cpr+ntotd;
	d_fs = d_jaco_f+ntotd;
	d_flx = d_fs+ntotd;
	 /*cudaMalloc(&d_nx, ntotd*sizeof(double));
        cudaMalloc(&d_ny, ntotd*sizeof(double));
        cudaMalloc(&d_nz, ntotd*sizeof(double));

        cudaMalloc(&d_rl, ntotd*sizeof(double));
        cudaMalloc(&d_ul, ntotd*sizeof(double));
        cudaMalloc(&d_wl, ntotd*sizeof(double));
        cudaMalloc(&d_vl, ntotd*sizeof(double));
        cudaMalloc(&d_pl, ntotd*sizeof(double));
        cudaMalloc(&d_tl, ntotd*sizeof(double));
        cudaMalloc(&d_al, ntotd*sizeof(double));
        cudaMalloc(&d_cpl, ntotd*sizeof(double));

        cudaMalloc(&d_rr, ntotd*sizeof(double));
        cudaMalloc(&d_ur, ntotd*sizeof(double));
        cudaMalloc(&d_vr, ntotd*sizeof(double));
        cudaMalloc(&d_wr, ntotd*sizeof(double));
        cudaMalloc(&d_pr, ntotd*sizeof(double));
        cudaMalloc(&d_tr, ntotd*sizeof(double));
        cudaMalloc(&d_ar, ntotd*sizeof(double));
        cudaMalloc(&d_cpr, ntotd*sizeof(double));*/

        cudaMalloc(&d_jaco_c, ntot*sizeof(double));
        /*cudaMalloc(&d_jaco_f, ntotd*sizeof(double));
        cudaMalloc(&d_fs, ntotd*sizeof(double));

        cudaMalloc(&d_flx, 5*ntotd*sizeof(double));*/

        //int* neq, int* nstate, int* nflux, int *nxd, int *nx1, int *nel, int *ndim, int *irho, int *iux, int *iuy, int *iuz, int *ipr, int *ithm, int *isnd, int *icpf, int *iph
        //printf("neq = %d, nxd = %d, nx1 = %d, nel = %d, ndim = %d, irho = %d\n",neq[0],nxd[0],nx1[0],nel[0],ndim[0],irho[0]);
        cudaError_t code = cudaPeekAtLastError();
        if (code != cudaSuccess){
            printf("cuda error Inv, malloc: %s\n",cudaGetErrorString(code));
        }

        

        InviscidFlux(d_qminus, d_qplus, d_flux, d_unx, d_uny, d_unz, d_area, d_wghtc, d_wghtf, d_cbc, d_jgl, d_jgt, d_nx, d_ny, d_nz, d_rl, d_ul, d_vl, d_wl, d_pl, d_tl, d_al, d_cpl, d_rr, d_ur, d_vr, d_wr, d_pr, d_tr, d_ar, d_cpr, d_fs, d_jaco_f, d_flx, d_jaco_c, neq[0], nstate[0], nflux[0], nxd[0], nx1[0], nel[0], ndim[0], irho[0]-1, iux[0]-1, iuy[0]-1, iuz[0]-1, ipr[0]-1, ithm[0]-1, isnd[0]-1, icpf[0]-1, iph[0]-1);
        code = cudaPeekAtLastError();
        if (code != cudaSuccess){
            printf("cuda error Inv, compute: %s\n",cudaGetErrorString(code));

        }

        
        cudaMemcpy(flux, d_flux, neq[0]*ntot*sizeof(double), cudaMemcpyDeviceToHost);
        //free

        cudaFree(d_qminus);
        cudaFree(d_qplus);
        cudaFree(d_flux);
        
        cudaFree(d_unx);
        cudaFree(d_uny);
        cudaFree(d_unz);
        
        cudaFree(d_area);

        cudaFree(d_wghtc);
        cudaFree(d_wghtf);
        
        cudaFree(d_cbc);//correct
        cudaFree(d_jgl);
        cudaFree(d_jgt);

       cudaFree(d_all); 
	/*cudaFree(d_nx);
        cudaFree(d_ny);
        cudaFree(d_nz);

        cudaFree(d_rl);
        cudaFree(d_ul);
        cudaFree(d_wl);
        cudaFree(d_vl);
        cudaFree(d_pl);
        cudaFree(d_tl);
        cudaFree(d_al);
        cudaFree(d_cpl);

        cudaFree(d_rr);
        cudaFree(d_ur);
        cudaFree(d_vr);
        cudaFree(d_wr);
        cudaFree(d_pr);
        cudaFree(d_tr);
        cudaFree(d_ar);
        cudaFree(d_cpr);*/

        cudaFree(d_jaco_c);
        /*cudaFree(d_jaco_f);
        cudaFree(d_fs);

        cudaFree(d_flx);*/

        

    }

    else{
        InviscidFlux(qminus, qplus, flux, unx, uny, unz, area, wghtc, wghtf, cbc, jgl, jgt, nx, ny, nz, rl, ul, vl, wl, pl, tl, al, cpl, rr, ur, vr, wr, pr, tr, ar, cpr, fs, jaco_f, flx, jaco_c, neq[0], nstate[0], nflux[0], nxd[0], nx1[0], nel[0], ndim[0], irho[0]-1, iux[0]-1, iuy[0]-1, iuz[0]-1, ipr[0]-1, ithm[0]-1, isnd[0]-1, icpf[0]-1, iph[0]-1);
    }
    
}

//res1 = vols
void before_fields(){

    //nekcopy u into res3 - can be done at next when res3 is needed
    //set_dealias_face without zwgl - can be done at next when wghtc, wghtf needed
    //compute_primitive_vars
    //fillq_gpu
    //faceu

}

__global__ void init_stokes(double *rpart, int *ipart, int nr, int ni, int n, int nw, int np, int nid, int jx, int jy, int jz, int jf0, int jar, int jai, double p){

    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < nw/np){
        int pid = id * np + nid + 1;
        double dumx = fmod(1.352 * id/7,0.98)+.01;
        double dumy = fmod(1.273 * id/8,0.98)+.01;
        double dumz = fmod(1.222 * id/9,0.98)+.01;
        int off = id*nr;
        rpart[off+jx] = -0.9 + dumx * 1.8;
        rpart[off+jy] = -0.9 + dumy * 1.8;
        rpart[off+jz] = -0.9 + dumz * 1.8;
        rpart[off+jf0] = 0.0;
        rpart[off+jar] = p;//pow(10,15);
        ipart[id*ni+jai] = pid;

    }
}

__global__ void particles_in_nid(int *fptsmap, double *rfpts, int *ifpts, double *rpart, int *ipart, double *range, int nrf, int nif, int *nfpts, int nr, int ni, int n, int lpart, int nelt, int jx, int jy, int jz,int je0, int jrc, int jpt, int jd, int jr, int nid){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < n){
        //double *rpart = rpart1 + id * nr;
        //int *ipart = ipart1 + id * ni;
        int ie;
        double xloc = rpart[id*nr+jx];
        double yloc = rpart[id*nr+jy];
        double zloc = rpart[id*nr+jz];
        for(ie = 0; ie < nelt; ie++){
            //double * range = xerange + ie * 6;
            if(xloc >= range[ie*6+0] && xloc <= range[ie*6+1] && yloc >=range[ie*6+2] && yloc <= range[ie*6+3] && zloc >= range[ie*6+4] && zloc <= range[ie*6+5]){
                ipart[id*ni+je0] = ie;
                ipart[id*ni+jrc] = 0;
                ipart[id*ni+jpt] = nid;
                ipart[id*ni+jd] = 1;
                rpart[id*nr+jr] = -1.0 + 2.0*(xloc-range[ie*6+0])/(range[ie*6+1]-range[ie*6+0]);
                rpart[id*nr+jr+1] = -1.0 + 2.0*(yloc-range[ie*6+2])/(range[ie*6+3]-range[ie*6+2]);
                rpart[id*nr+jr] = -1.0 + 2.0*(zloc-range[ie*6+4])/(range[ie*6+5]-range[ie*6+4]);
                break;
            }
        }
        if(ie==nelt){
            //point is outside all elements
            int old = atomicAdd(nfpts, 1);
            if(old==lpart){
                printf("error many moving particles\n");
                return;
            }
            fptsmap[old] = id+1;
            //double * rfp = rfpts + old * nrf;
            //int * ifp = ifpts + old * nif;
            for(int i = 0 ; i < nrf; i++)
                rfpts[old*nrf+i] = rpart[id*nr+i];

            for(int i = 0 ; i < nif; i++)
                ifpts[old*nif+i] = ipart[id*ni+i];
        }
    }

}

extern "C" void particles_in_nid_wrapper_(double *rfpts, int *ifpts, double *rpart, int *ipart, double *xerange, int *fptsmap, int *nrf, int *nif, int *nfpts, int *nr, int *ni, int *n, int *lpart, int *nelt, int *jx, int *jy, int *jz,int *je0, int *jrc, int *jpt, int *jd, int *jr, int *nid){

    float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    bool inCPU = false;
    double *d_rfpts, *d_rpart, *d_xerange;
    int *d_fptsmap, *d_ifpts, *d_ipart, *d_nfpts;
    if(inCPU){
        cudaMalloc(&d_rfpts, lpart[0]*nrf[0]*sizeof(double));
        cudaMalloc(&d_rpart, n[0]*nr[0]*sizeof(double));
        cudaMalloc(&d_xerange, nelt[0]*6*sizeof(double));
        cudaMalloc(&d_fptsmap, lpart[0]*sizeof(int));
        cudaMalloc(&d_ifpts, lpart[0]*nif[0]*sizeof(int));
        cudaMalloc(&d_ipart, n[0]*ni[0]*sizeof(int));
        
        cudaMalloc(&d_nfpts, sizeof(int));


        cudaMemcpy(d_rpart, rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_xerange, xerange, nelt[0]*6*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ipart, ipart, n[0]*ni[0]*sizeof(int), cudaMemcpyHostToDevice);


        cudaMemcpy(d_nfpts, nfpts, sizeof(int), cudaMemcpyHostToDevice);


    }
    else{
        d_rfpts = rfpts;
        d_rpart= rpart;
        d_xerange = xerange;
        d_fptsmap = fptsmap;
        d_ifpts = ifpts;
        d_ipart = ipart;
        cudaMalloc(&d_nfpts, sizeof(int));
        cudaMemcpy(d_nfpts, nfpts, sizeof(int), cudaMemcpyHostToDevice);

    }
    int blockSize = 1024, gridSize;
    gridSize = (int)ceil((float)n[0]/blockSize);
    // printf ("print var %d %d %d\n", n[0], jx[0], jy[0]);
    particles_in_nid<<<gridSize, blockSize>>>(d_fptsmap, d_rfpts, d_ifpts, d_rpart, d_ipart, d_xerange, nrf[0], nif[0], d_nfpts, nr[0], ni[0], n[0], lpart[0], nelt[0], jx[0]-1, jy[0]-1, jz[0]-1, je0[0]-1, jrc[0]-1, jpt[0]-1, jd[0]-1, jr[0]-1, nid[0]);
    if(inCPU){
        cudaMemcpy(ipart, d_ipart, n[0]*ni[0]*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(rpart, d_rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(nfpts, d_nfpts, sizeof(int), cudaMemcpyDeviceToHost);

        if(nfpts[0]>0){
            cudaMemcpy(fptsmap, d_fptsmap, nfpts[0]*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(rfpts, d_rfpts, nfpts[0]*nrf[0]*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(ifpts, d_ifpts, nfpts[0]*nif[0]*sizeof(int), cudaMemcpyDeviceToHost);

        }
        //free
	cudaFree(d_rpart);
        cudaFree(d_ipart);
        cudaFree(d_xerange);
        cudaFree(d_fptsmap);
        cudaFree(d_rfpts);
        cudaFree(d_ifpts);
    }
    else{
        cudaMemcpy(nfpts, d_nfpts, sizeof(int), cudaMemcpyDeviceToHost);
        // printf ("print var 1st %d\n", nfpts);
    }
    cudaFree(d_nfpts);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    // printf ("print var 2nd %d\n", nfpts);
    //printf("particles in nid time is %f\n",time*1e-03);

}

extern "C" void init_stokes_particleswrapper_(double *rpart, int *ipart, int *nr, int *ni, int *n, int *nw, int *np, int *nid, int *jx, int *jy, int *jz, int *jf0, int *jar, int *jai){
    bool inCPU = false;
    double *d_rpart;
    int *d_ipart;
    if(inCPU){
        cudaMalloc(&d_rpart, n[0]*nr[0]*sizeof(double));
        cudaMalloc(&d_ipart, n[0]*ni[0]*sizeof(int));

    }
    else{
        d_rpart = rpart;
        d_ipart = ipart;
    }
    int blockSize = 1024, gridSize;
    int proc_work = nw[0]/np[0];
    gridSize = (int)ceil((float)proc_work/blockSize);
    init_stokes<<<gridSize, blockSize>>>(rpart, ipart, nr[0], ni[0], n[0], nw[0], np[0], nid[0], jx[0]-1, jy[0]-1, jz[0]-1, jf0[0]-1, jar[0]-1, jai[0]-1, pow(10,15));
    if(inCPU){
        cudaMemcpy(rpart, d_rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(ipart, d_ipart, n[0]*ni[0]*sizeof(int), cudaMemcpyDeviceToHost);
        //free

        cudaFree(d_rpart);
        cudaFree(d_ipart);

    }
}

__global__ void solve_velocity(double *rpart, int nr, int ni, int n, int j, int jx0, int jx1, int jx2, int jx3, int jv0, int jv1, int jv2, int jv3, int ju0, int ju1, int ju2, int ju3){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < n){
        int off = id*nr+j;
        double * rpart_off = rpart+off;
        rpart_off[ju3] = rpart_off[ju2];
        rpart_off[ju2] = rpart_off[ju1];
        rpart_off[ju1] = rpart_off[ju0];
        rpart_off[jv3] = rpart_off[jv2];
        rpart_off[jv2] = rpart_off[jv1];
        rpart_off[jv1] = rpart_off[jv0];
        rpart_off[jx3] = rpart_off[jx2];
        rpart_off[jx2] = rpart_off[jx1];
        rpart_off[jx1] = rpart_off[jx0];
    }

}

__global__ void update_velocity(double *rpart1, double *alpha, double *beta, int ndim, int nr, int ni, int n, int jx0, int jx1, int jx2, int jx3, int jv0, int jv1, int jv2, int jv3, int ju0, int ju1, int ju2, int ju3, int jf0, int jst){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < n*ndim){
        int j = id%ndim;
        int i = id/ndim;
        double s = rpart1[i*nr+jst];
        int off = i*nr+j;
        double * rpart = rpart1+off;
        double rhs = s*(alpha[1]*rpart[ju1] + alpha[2]*rpart[ju2] + alpha[3]*rpart[ju3]) + rpart[jf0-j] + beta[1]*rpart[jv1] + beta[2]*rpart[jv2] + beta[3]*rpart[jv3];
        rpart[jv0] = rhs/(beta[0]+s);
        double rhx = beta[1]*rpart[jx1] + beta[2]*rpart[jx2] + beta[3]*rpart[jx3] + rpart[jv0];
        // rpart[jx0] = rhx/beta[0];
    }
}

__global__ void update_particle_location(double *rpart1, double *xdrange1, int nr, int n, int ndim, int jx0, int jx1, int jx2, int jx3, int jaa, int jab, int jac, int jad, int *flagsend, double dt){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
//   printf("***update_particle_location %d\n", id);
    if(id < n*ndim){
//    if(id < n){
//        printf("***entered if block\n");
        int j = id%ndim;
        int i = id/ndim;
        int off = i*nr+j;
        double * rpart = rpart1+off;
        double *xdrange = xdrange1+2*j;

//      curtain test case, update x location
/*        int factor=(3-j)/3;
        rpart[jx0] = rpart[jx0] + (1.0/3)*rpart[jx0]*dt*factor;
*/

//      new curtain test case, update y location
        int factor = 0;
        if (j == 1){
           factor = 1;
        }
        rpart[jx0] = rpart[jx0] + (1.0/3)*rpart[jx0]*dt*factor;

#if 0
        // Do not remove this part of the code
        rpart[jx0] = rpart[jx0] + rpart[jaa] * rpart[jad-j];
#endif
        if(rpart[jx0]<xdrange[0]){
            //exit(0);
            flagsend[0] = flagsend[0] + 1;
        }
        if(rpart[jx0] > xdrange[1]){
            //exit(0);
            flagsend[0] = flagsend[0] + 1;
        }
#if 0
        // Do not remove this part of the code
        if(rpart[jx0]<xdrange[0]){
            rpart[jx0] = xdrange[1] - fabs(xdrange[0] - rpart[jx0]);
            rpart[jx1] = xdrange[1] + fabs(xdrange[0] - rpart[jx1]);
            rpart[jx2] = xdrange[1] + fabs(xdrange[0] - rpart[jx2]);
            rpart[jx3] = xdrange[1] + fabs(xdrange[0] - rpart[jx3]);
        }
        if(rpart[jx0] > xdrange[1]){
            rpart[jx0] = xdrange[0] + fabs(rpart[jx0] - xdrange[1]);
            rpart[jx1] = xdrange[0] - fabs(xdrange[2] - rpart[jx1]);
            rpart[jx2] = xdrange[0] - fabs(xdrange[2] - rpart[jx2]);
            rpart[jx3] = xdrange[0] - fabs(xdrange[2] - rpart[jx3]);

        }
#endif

    }
}

//not use this function
__global__ void update_particle_location_keke(double *rpart1, double *xdrange1, int nr, int n, int j, int ndim, int jx0, int jx1, int jx2, int jx3, int jaa, int jab, int jac, int jad){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
//   printf("***update_particle_location %d\n", id);
    if(id < n){
//        printf("***entered if block\n");
//        int j = id%ndim;
//        int i = id/ndim;
        int off = id*nr;
        double * rpart = rpart1+off;
        double *xdrange = xdrange1+2*j;
        rpart[jx0+j] = rpart[jx0+j] + rpart[jaa+j] * rpart[jad];
//        rpart[jx0+1] = rpart[jx0+1] + rpart[jab] * rpart[jad];
//        rpart[jx0+2] = rpart[jx0+2] + rpart[jac] * rpart[jad];
        rpart = rpart + j;     //avoid the following all +j
        if(rpart[jx0]<xdrange[0]){
            rpart[jx0] = xdrange[1] - fabs(xdrange[0] - rpart[jx0]);
            rpart[jx1] = xdrange[1] + fabs(xdrange[0] - rpart[jx1]);
            rpart[jx2] = xdrange[1] + fabs(xdrange[0] - rpart[jx2]);
            rpart[jx3] = xdrange[1] + fabs(xdrange[0] - rpart[jx3]);
        }
        if(rpart[jx0] > xdrange[1]){
            rpart[jx0] = xdrange[0] + fabs(rpart[jx0] - xdrange[1]);
            rpart[jx1] = xdrange[0] - fabs(xdrange[2] - rpart[jx1]);
            rpart[jx2] = xdrange[0] - fabs(xdrange[2] - rpart[jx2]);
            rpart[jx3] = xdrange[0] - fabs(xdrange[2] - rpart[jx3]);

        }

    }
}
void update_stokes_particles(double *rpart, double *alpha, double *beta, double *xdrange, int ndim, int nr, int ni, int n, int jx0, int jx1, int jx2, int jx3, int jv0, int jv1, int jv2, int jv3, int ju0, int ju1, int ju2, int ju3, int jar, int jf0, int jaa, int jab, int jac, int jad, int *flagsend, double dt){
    //jx0, ... all should be passed original-1
    //alpha,beta[0:3]
    //solve velocity
    int blockSize = 1024, gridSize;
    gridSize = (int)ceil((float)n/blockSize);
    for(int j = 0; j < ndim; j++)
        solve_velocity<<<gridSize, blockSize>>>(rpart, nr, ni, n, j, jx0, jx1, jx2, jx3, jv0, jv1, jv2, jv3, ju0, ju1, ju2, ju3);
    
    gridSize = (int)ceil((float)n*ndim/blockSize);
    update_velocity<<<gridSize, blockSize>>>(rpart, alpha, beta, ndim, nr, ni, n, jx0, jx1, jx2, jx3, jv0, jv1, jv2, jv3, ju0, ju1, ju2, ju3, jf0, jar);

//    gridSize = (int)ceil((float)n/blockSize);
//    for(int j = 0; j < ndim; j++)
//        update_particle_location_keke<<<gridSize, blockSize>>>(rpart, xdrange, nr, n, j, ndim, jx0, jx1, jx2, jx3, jaa, jab, jac, jad);
    update_particle_location<<<gridSize, blockSize>>>(rpart, xdrange, nr, n, ndim, jx0, jx1, jx2, jx3, jaa, jab, jac, jad, flagsend, dt); //previous one with gridSize=..*ndim

    
}

extern "C" void updatestokeswrapper_(double *rpart, double *alpha, double *beta, double *xdrange, int *ndim, int *nr, int *ni, int *n, int *jx0, int *jx1, int *jx2, int *jx3, int *jv0, int *jv1, int *jv2, int *jv3, int *ju0, int *ju1, int *ju2, int *ju3, int *jar, int *jf0, int *jaa, int *jab, int * jac, int * jad, int *flagsend, double* dt){

    float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    bool inCPU = false;
    double * d_rpart, *d_alpha, *d_beta, *d_xdrange;
    int *d_flagsend;
    if(inCPU){
        cudaMalloc(&d_rpart, n[0]*nr[0]*sizeof(double));
        cudaMalloc(&d_alpha, 4*sizeof(double));
        cudaMalloc(&d_beta, 4*sizeof(double));
        cudaMalloc(&d_xdrange, 6*sizeof(double));
        cudaMalloc(&d_flagsend, sizeof(int));
        cudaMemcpy(d_rpart, rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_alpha, alpha, 4*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, beta, 4*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_xdrange, xdrange, 6*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_flagsend, flagsend, sizeof(int), cudaMemcpyHostToDevice);
    }
    else{
	d_rpart = rpart;
        //d_alpha = alpha;
        //d_beta = beta;
        d_xdrange = xdrange;
        cudaMalloc(&d_alpha, 4*sizeof(double));
        cudaMalloc(&d_beta, 4*sizeof(double));
        cudaMemcpy(d_alpha, alpha, 4*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, beta, 4*sizeof(double), cudaMemcpyHostToDevice);
        d_flagsend = flagsend;
    }
    update_stokes_particles(d_rpart, d_alpha, d_beta, d_xdrange, ndim[0], nr[0], ni[0], n[0], jx0[0]-1, jx1[0]-1, jx2[0]-1, jx3[0]-1, jv0[0]-1, jv1[0]-1, jv2[0]-1, jv3[0]-1, ju0[0]-1, ju1[0]-1, ju2[0]-1, ju3[0]-1, jar[0]-1, jf0[0]-1, jaa[0]-1, jab[0]-1, jac[0]-1, jad[0]-1, d_flagsend, dt[0]);

    if(inCPU){
        cudaMemcpy(rpart, d_rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyDeviceToHost);
        //free

        cudaFree(d_rpart);
        //cudaFree(d_alpha);
        //cudaFree(d_beta);
        cudaFree(d_xdrange);
        cudaFree(d_flagsend);
    }
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    //printf("update stokes time is %f\n",time*1e-03);

}

__global__ void baryinterp(double *rpart, int *ipart, double *vx, double *vy, double *vz, int jr, int je0, int ju0, double *rep, double *xgll, double * ygll, double *zgll, double *wxgll, double *wygll, double *wzgll, int nx1, int n, int nr, int ni){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < n){
        int x,y,z;
        double bot = 0.0;
        x = rpart[id*nr+jr];
        y = rpart[id*nr+jr+1];
        z = rpart[id*nr+jr+2];
        for(int k=0; k<nx1;k++){
            for(int j=0; j<nx1; j++){
                double repdum = wygll[j]/(y-ygll[j]) * wzgll[k]/(z-zgll[k]) ;
                for(int i = 0; i<nx1; i++){
                    rep[k*nx1*nx1+j*nx1+i] = repdum * wxgll[i]/(x-xgll[i]);
                    bot = bot + rep[k*nx1*nx1+j*nx1+i];
                }
            }
        }

        int ie = ipart[id*ni+je0];
        //new (vx(ie),rpart(ju0)
        double top1 = 0.0, top2 = 0.0, top3 = 0.0;
        int nxyz = nx1*nx1*nx1;
        double *fieldx = vx+ie*nxyz;
        double *fieldy = vy+ie*nxyz;
        double *fieldz = vz+ie*nxyz;
        for(int i = 0; i<nxyz; i++){
            top1 = top1 + rep[i]*fieldx[i];
            top2 = top2 + rep[i]*fieldy[i];
            top3 = top3 + rep[i]*fieldz[i];
        }
        rpart[id*nr+ju0] = top1/bot;
        rpart[id*nr+ju0+1] = top2/bot;
        rpart[id*nr+ju0+2] = top3/bot;


    }

}

extern "C" void baryweights_evalwrapper_(double *rpart, int *ipart, double *vx, double *vy, double *vz, double *rep, double *xgll, double * ygll, double *zgll, double *wxgll, double *wygll, double *wzgll, int* jr, int* je0, int* ju0,  int* nx1, int* n, int* nr, int* ni, int *nel){

    float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    bool inCPU = false;
    double *d_rpart, *d_vols, *d_rep, *d_gll;
    int *d_ipart;
    int nx1_2 = nx1[0]*nx1[0];
    int nx1_3 = nx1_2*nx1[0];

     if(inCPU){
        cudaMalloc(&d_rpart, n[0]*nr[0]*sizeof(double));
        cudaMalloc(&d_vols, 3*nel[0]*nx1_3*sizeof(double));
        cudaMalloc(&d_rep, nx1_3*sizeof(double));
        cudaMalloc(&d_gll, 6*nx1[0]*sizeof(double));
        cudaMalloc(&d_ipart, n[0]*ni[0]*sizeof(int));

        cudaMemcpy(d_rpart, rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vols, vx, nel[0]*nx1_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vols+nel[0]*nx1_3, vy, nel[0]*nx1_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vols+2*nel[0]*nx1_3, vz, nel[0]*nx1_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rep, rep, nx1_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gll, xgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gll+nx1[0], ygll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gll+2*nx1[0], zgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gll+3*nx1[0], wxgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gll+4*nx1[0], wygll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gll+5*nx1[0], wzgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_ipart, ipart, n[0]*ni[0]*sizeof(int), cudaMemcpyHostToDevice);
        int blockSize = 1024, gridSize;
        gridSize = (int)ceil((float)n[0]/blockSize);
        baryinterp<<<gridSize, blockSize>>>(d_rpart, d_ipart, d_vols, d_vols+nel[0]*nx1_3, d_vols+2*nel[0]*nx1_3, jr[0]-1, je0[0]-1, ju0[0]-1, d_rep, d_gll, d_gll+nx1[0], d_gll+2*nx1[0], d_gll+3*nx1[0], d_gll+4*nx1[0], d_gll+5*nx1[0], nx1[0], n[0], nr[0], ni[0]);


    }
    else{
        d_rpart = rpart;
        d_ipart = ipart;
        cudaMalloc(&d_rep, nx1_3*sizeof(double));
        cudaMalloc(&d_gll, 6*nx1[0]*sizeof(double));
        cudaMemcpy(d_rep, rep, nx1_3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gll, xgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gll+nx1[0], ygll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gll+2*nx1[0], zgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gll+3*nx1[0], wxgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gll+4*nx1[0], wygll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gll+5*nx1[0], wzgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
        int blockSize = 1024, gridSize;
        gridSize = (int)ceil((float)n[0]/blockSize);
        baryinterp<<<gridSize, blockSize>>>(d_rpart, d_ipart, vx,vy,vz, jr[0]-1, je0[0]-1, ju0[0]-1, d_rep, d_gll, d_gll+nx1[0], d_gll+2*nx1[0], d_gll+3*nx1[0], d_gll+4*nx1[0], d_gll+5*nx1[0], nx1[0], n[0], nr[0], ni[0]);
    }
    if(inCPU){
        cudaMemcpy(rpart, d_rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyDeviceToHost);
        //free

        cudaFree(d_rpart);
        cudaFree(d_vols);
        cudaFree(d_rep);
        cudaFree(d_gll);
        cudaFree(d_ipart);
    }
    else{
        cudaFree(d_rep);
        cudaFree(d_gll);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    //printf("interp time is %f\n",time*1e-03);

}

__global__ void packFaces(double *faces, double *packed, double *sharedIndex, int n, int nelt, int nx1, int iu1, int dir){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
    	if(id < n){
		//get element and face numbers
		int e, f, nx1_2;
		nx1_2 = nx1 * nx1;
		e = sharedIndex[id*2]-1;//fmod(sharedIndex[id*2]-1,nelt);
		f = sharedIndex[id*2+1]-1;
		//printf("e = %d, f = %d\n",e,f);
		/*if(e>nelt-1)
			printf ("e  > nelt, %d\n",e);
		if(f>5)
			printf("f > nface,%d\n",f);*/
		//copy the whole face
		int off2 = id * nx1_2;
		int f_off2 = e * 6 * nx1_2 + f*nx1_2;
		for(int i = 0; i < 5; i++){
			int off1 = i * n * nx1_2;
			int f_off1 = i * nelt * 6 * nx1_2;
			double* packed1 = packed+off1+off2;
			double* faces1 = faces+f_off1+f_off2;
			for(int j = 0; j < nx1_2; j++){
				if(dir == 0)
					packed1[j] = faces1[j];
				else faces1[j] = packed1[j];
			}
		}
		for(int i = 0; i < 5; i++){
                        int off1 = (i+5) * n * nx1_2;
                        int f_off1 = (i+iu1-1) * nelt * 6 * nx1_2;
                        double* packed1 = packed+off1+off2;
                        double* faces1 = faces+f_off1+f_off2;
                        for(int j = 0; j < nx1_2; j++){
                                if(dir == 0)
                                        packed1[j] = faces1[j];
				else faces1[j] = packed1[j];
                        }
                }

	}
}
extern "C" void packfaceswrapper_(double *faces, double *packed, double *sharedIndex, int *maxIndex, int *nelt, int *nx1, int *iu, int *dir){

	//all data is in GPU
	int blockSize = 1024, gridSize;
	float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
 	double *d_shared;
	cudaMalloc(&d_shared, nelt[0]*12*sizeof(double));

        cudaMemcpy(d_shared, sharedIndex, nelt[0]*12*sizeof(double), cudaMemcpyHostToDevice);
	cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    // printf("allocate sharedIndex is %f\n",time*1e-03);
    	gridSize = (int)ceil((float)maxIndex[0]/blockSize);
    	packFaces<<<gridSize, blockSize>>>(faces, packed, d_shared, maxIndex[0], nelt[0], nx1[0], iu[0], dir[0]);
	cudaFree(d_shared);
	cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess){
        printf("cuda error str 4: %s\n",cudaGetErrorString(code));
    }
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
    //printf("second element is %f, n %d, k%d, time is %f\n",mat[n*k+1],n,k,time*1e-03);
                                         


    //do in cpu
    cudaEventRecord(startEvent, 0);
    for(int i =0; i< n*k*jobs; i++)
        mat[i] = u_eq1[i];
    for(int i=0; i< n*k*jobs;i++)
        mat[i] = mat[i] * vx[i];
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    //printf("cpu time is %f\n",time*1e-03);

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

	//cuda_multi_gemm_unif(stream, 'N', 'N', dim, dim, dim, &alpha, dim, dim*dim, d_A, d_B, d_BB, &beta, d_C, d_D, d_E, jobs*K, gridsize);
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
        cudaStreamDestroy(stream);

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

