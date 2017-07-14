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

__global__ void particles_in_nid(int *fptsmap, double *rfpts, int *ifpts, double *rpart, int *ipart, double *range, int nrf, int nif, int nfpts, int nr, int ni, int n, int lpart, int nelt, int jx, int jy, int jz,int je0, int jrc, int jpt, int jd, int jr, int nid){
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
                rpart[id*nr+jr+2] = -1.0 + 2.0*(zloc-range[ie*6+4])/(range[ie*6+5]-range[ie*6+4]);
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


__global__ void update_particle_location(double *rpart1, int *ipart, double *xdrange1, int *in_part, int *bc_part, int ndim, int nr, int ni, int n, int jx, int jx1, int jx2, int jx3){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < n*ndim){
          int i = id/ndim;
          int j = id%ndim;
          int off = i*nr+j;
          double *rpart = rpart1+off;
          double *xdrange = xdrange1+2*j;

          if (rpart[jx] < xdrange[0]){
                if ( (bc_part[1] == 0 && j == 0) || (bc_part[3] == 0 && j == 1) || (bc_part[5] == 0 && j == 2) ){
                     rpart[jx] = xdrange[1] - fabs(xdrange[0] - rpart[jx]);
                     rpart[jx1] = xdrange[1] + fabs(xdrange[0] - rpart[jx1]);
                     rpart[jx2] = xdrange[1] + fabs(xdrange[0] - rpart[jx2]);
                     rpart[jx3] = xdrange[1] + fabs(xdrange[0] - rpart[jx3]);
               }
               else if ( (bc_part[1] != 0 && j == 0) || (bc_part[3] != 0 && j == 1) || (bc_part[5] != 0 && j == 2) ){
                     int old = atomicExch(in_part[i], -1);
               }
         }
         if (rpart[jx] > xdrange[1]){
               if ( (bc_part[1] == 0 && j == 0) || (bc_part[3] == 0 && j == 1) || (bc_part[5] == 0 && j == 2) ){
                    rpart[jx] = xdrange[0] + fabs(xdrange[0] - rpart[jx]);
                    rpart[jx1] = xdrange[0] - fabs(xdrange[0] - rpart[jx1]);
                    rpart[jx2] = xdrange[0] - fabs(xdrange[0] - rpart[jx2]);
                    rpart[jx3] = xdrange[0] - fabs(xdrange[0] - rpart[jx3]);
              }
              else if ( (bc_part[1] != 0 && j == 0) || (bc_part[3] != 0 && j == 1) || (bc_part[5] != 0 && j == 2) ){
                    int old = atomicExch(in_part[i], -1);
              }

         }
   }
}



// __global__ void update_data_if_outflow(double *rpart1, int *ipart1, double *in_part, int *in_part, int ic, int nr, int ir){
//       int id = blockIdx.x*blockDim.x+threadIdx.x;
//       if(id < n*ndim){
//             double *rpart = rpart1+(id*nr);
//             double *ipart = ipart1+(id*ir);
//             int sum = abs(bc_part[1])+abs(bc_part[2])+abs(bc_part[3])+abs(bc_part[4])+abs(bc_part[5])+abs(bc_part[6]);
//             if (sum > 0){
//                   if (in_part[id] == 0){
//                         int old = atomicAdd(nfpts, 1);
//                         for(int k=0;k < nr; k++){
//
//                         }
//                   }
//             }
//       }
//
// }


extern "C" void particles_in_nid_wrapper_(int *fptsmap, double *rfpts, int *ifpts, double *rpart, int *ipart, double *xerange, int *nrf, int *nif, int *nfpts, int *nr, int *ni, int *n, int *lpart, int *nelt, int *jx, int *jy, int *jz,int *je0, int *jrc, int *jpt, int *jd, int *jr, int *nid){

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
//        if(nfpts[0]>0){
//            cudaMemcpy(fptsmap, d_fptsmap, nfpts[0]*sizeof(int), cudaMemcpyDeviceToHost);
//            cudaMemcpy(rfpts, d_rfpts, nfpts[0]*nrf[0]*sizeof(double), cudaMemcpyDeviceToHost);
//            cudaMemcpy(ifpts, d_ifpts, nfpts[0]*nif[0]*sizeof(int), cudaMemcpyDeviceToHost);

//        }
        // printf ("print var 1st %d\n", nfpts);
    }
    cudaFree(d_nfpts);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    // printf ("print var 2nd %d\n", nfpts);
    //printf("particles in nid time is %f\n",time*1e-03);

}

extern "C" void update_particle_location_wrapper_(double *rpart, int *ipart, double *xdrange, int *in_part, int *bc_part, int *ndim, int *nr, int *ni, int *n, int *jx, int *jx1, int *jx2,int *jx3){

    float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    bool inCPU = false;
    double *d_rpart, *d_xdrange;
    int *d_ipart, *d_in_part;
    int ic = 0;
    if(inCPU){

        cudaMalloc(&d_rpart, n[0]*nr[0]*sizeof(double));
        cudaMalloc(&d_xdrange, 6*sizeof(double));
        cudaMalloc(&d_in_part, n[0]*sizeof(int));
        cudaMalloc(&d_ipart, n[0]*ni[0]*sizeof(int));
        cudaMemcpy(d_rpart, rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_xerange, xerange, nelt[0]*6*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ipart, ipart, n[0]*ni[0]*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_in_part, in_part, n[0]*sizeof(int), cudaMemcpyHostToDevice);


  }
  else{
      d_rpart= rpart;
      d_xdrange = xdrange;
      d_in_part = in_part;
      d_ipart = ipart;
      d_bc_part = bc_part;
      cudaMalloc(&d_in_part, n[0]*sizeof(int));
      cudaMemcpy(d_in_part, in_part, n[0]*sizeof(int), cudaMemcpyHostToDevice);

  }
  int blockSize = 1024, gridSize;
  gridSize = (int)ceil((float)n[0]*ndim[0]/blockSize);
  update_particle_location<<<gridSize, blockSize>>>(d_rpart,d_ipart,d_xdrange,d_inpart,d_bc_part,ndim[0],nr[0],ni[0],n[0],jx[0],jx1[0],jx2[0],jx3[0]);
//----------need to confirm with Dr. Tania and Dr. Ranka if there is a way to parallelize the memory update - right now im implementing it in the host
//  gridSize = (int)ceil((float)n[0]/blockSize);
//  updated_data_if_outflow<<<gridSize, blockSize>>>(d_rpart,s_ipart,d_inpart,&ic);
  if(inCPU){
      cudaMemcpy(ipart, d_ipart, n[0]*ni[0]*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(rpart, d_rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(in_part, d_in_part, n[0]*sizeof(int), cudaMemcpyDeviceToHost);


      //free
      cudaFree(d_rpart);
      cudaFree(d_ipart);
      cudaFree(d_xdrange);
      cudaFree(d_in_part);
  }
  else{
      cudaMemcpy(in_part, d_in_part, n[0]*sizeof(int), cudaMemcpyDeviceToHost);
//        if(nfpts[0]>0){
//            cudaMemcpy(fptsmap, d_fptsmap, nfpts[0]*sizeof(int), cudaMemcpyDeviceToHost);
//            cudaMemcpy(rfpts, d_rfpts, nfpts[0]*nrf[0]*sizeof(double), cudaMemcpyDeviceToHost);
//            cudaMemcpy(ifpts, d_ifpts, nfpts[0]*nif[0]*sizeof(int), cudaMemcpyDeviceToHost);

//        }
      // printf ("print var 1st %d\n", nfpts);
  }
  cudaFree(d_in_part);
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&time, startEvent, stopEvent);
  // printf ("print var 2nd %d\n", nfpts);
  //printf("particles in nid time is %f\n",time*1e-03);

}
