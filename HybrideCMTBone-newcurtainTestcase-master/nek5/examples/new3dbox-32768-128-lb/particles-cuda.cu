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


__global__ void update_particle_location(double *rpart1, double *xdrange1, int *in_part, int *bc_part, int n, int ndim, int nr, int jx, int jx1, int jx2, int jx3){
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


__global__ void interp_props_part_location(double *rpart, int *ipart, double *vx,double *vy,double *vz,double *t,double *vtrans,double *rep,double *xgll,double *ygll,double *zgll,double *wxgll,double *wygll,double *wzgll,int n, int nr, int ni, int nx1, int nx1r, int jr, int ju0, int je0, int jtemp, int jrho){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n){
      double x,y,z;
      double bot = 0.0;
      double diff;
      double repdum;
      double bwgtz[nx1];
      double bwgty[nx1];
      double bwgtx[nx1];
      x = rpart[id*nr+jr];
      y = rpart[id*nr+jr+1];
      z = rpart[id*nr+jr+2];
  // barycentric interpolation initialization
      for (int k = 0; k < nx1r; k++){
            diff = z - zgll[k];
            if (fabs(diff) < 1.0e-16){
                  if (diff > 0){
                        diff = 1.0e-16;
                  }
                  else{
                        diff = -1.0e-16;
                  }

            }
            bwgtz[k] = wzgll[k]/diff;
     }
     for (int i = 0; i < nx1r; i++){
           diff = x - xgll[i];
           if (fabs(diff) < 1.0e-16){
                 if (diff > 0){
                       diff = 1.0e-16;
                 }
                 else{
                       diff = -1.0e-16;
                 }

           }
           bwgtx[i] = wxgll[i]/diff;
    }
    for (int j = 0; j < nx1r; j++){
          diff = y - ygll[j];
          if (fabs(diff) < 1.0e-16){
               if (diff > 0){
                      diff = 1.0e-16;
               }
               else{
                      diff = -1.0e-16;
               }

          }
          bwgty[j] = wygll[j]/diff;
   }

   for(int k=0; k<nx1r;k++){
         for(int j=0; j<nx1r; j++){
             repdum = bwgty[j] * bwgtz[k] ;
             for(int i = 0; i<nx1r; i++){
                 rep[k*nx1r*nx1r+j*nx1r+i] = repdum * bwgtx[i];
                 bot = bot + rep[k*nx1*nx1+j*nx1+i];
             }
         }
     }
  for(int k=0; k<nx1r;k++){
        for(int j=0; j<nx1r; j++){
              for(int i = 0; i<nx1r; i++){
                    rep[k*nx1r*nx1r+j*nx1r+i] = rep[k*nx1r*nx1r+j*nx1r+i]/bot;
              }

        }

  }

  // barycentric interpolation initialization ends  //

  int ie = ipart[id*ni+je0];
  double top1 = 0.0, top2 = 0.0, top3 = 0.0, top4 = 0.0, top5 = 0.0;
  int nxyz = nx1*nx1*nx1;
  double *fieldx = vx+ie*nxyz;
  double *fieldy = vy+ie*nxyz;
  double *fieldz = vz+ie*nxyz;
  double *fieldt = t+ie*nxyz;
  double *fieldvtrans = vtrans+ie*nxyz;
  //full interpolation
  if (nx1r == nx1){
        for (int i=0; i< nxyz; i++){
              top1 = top1 + rep[i]*fieldx[i];
              top2 = top2 + rep[i]*fieldy[i];
              top3 = top3 + rep[i]*fieldz[i];
              top4 = top4 + rep[i]*fieldt[i];
              top5 = top5 + rep[i]*fieldvtrans[i];
        }
        rpart[id*nr+ju0] = top1;
        rpart[id*nr+ju0+1] = top2;
        rpart[id*nr+ju0+2] = top3;
        rpart[id*nr+jtemp] = top4;
        rpart[id*nr+jrho] = top5;
  }

  else{
        // reduced barycentric interpolation
        int kk = 0,jj = 0, ii = 0;
        double ijk3,ijk2,ijk1;
        for (int k=0;k < nx1; k+=2){
              for (int j=0;j<nx1;j+=2){
                    for (int i=0;i<nx1;i+=2){
                          top1 = top1+rep[k*nx1*nx1+j*nx1+i]*fieldx[k*nx1*nx1+j*nx1+i];
                          top2 = top2+rep[k*nx1*nx1+j*nx1+i]*fieldy[k*nx1*nx1+j*nx1+i];
                          top3 = top3+rep[k*nx1*nx1+j*nx1+i]*fieldz[k*nx1*nx1+j*nx1+i];
                          top4 = top4+rep[k*nx1*nx1+j*nx1+i]*fieldx[k*nx1*nx1+j*nx1+i];
                          top5 = top5+rep[k*nx1*nx1+j*nx1+i]*fieldx[k*nx1*nx1+j*nx1+i];
                    }
              }
        }
        rpart[id*nr+ju0] = top1;
        rpart[id*nr+ju0+1] = top2;
        rpart[id*nr+ju0+2] = top3;
        rpart[id*nr+jtemp] = top4;
        rpart[id*nr+jrho] = top5;

  }

  }

}

//-----------------------------------------------------------

__global__ void usr_particles_forces_bdf(double *rpart, int n, int nr,int jvol,int jrhop, int jfusr, int jf0,int jrho,int jfqs,int ju0,int jv0, int ja, int jdp, int jre, int jtaup, int jcd){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < n){
//          int i = id/ndim;
//          int j = id%ndim;
          double pmass,pmassf,rdum;
          double uvel[3],vvel[3];
          double c0,T0,mu_p,Re_p,M_p,vel_diff;
          pmass = rpart[id*nr+jvol]*rpart[id*nr+jrhop];
          //compute_re_particles
          uvel[0] = rpart[id*nr+ju0];
          uvel[1] = rpart[id*nr+ju0+1];
          uvel[2] = rpart[id*nr+ju0+2];
          vvel[0] = rpart[id*nr+jv0];
          vvel[1] = rpart[id*nr+jv0+1];
          vvel[2] = rpart[id*nr+jv0+2];
          c0 = 120.0;
          T0 = 291.15;
          c_sound = 1000000; //fix this at the host level so that no concurrency issues arise
          rpart[id*nr+ja] = c_sound;
          mu_p = mu_0;
          vel_diff = sqrt(((uvel[0]-vvel[0])*(uvel[0]-vvel[0]))+((uvel[0]-vvel[0])*(uvel[1]-vvel[1]))+((uvel[2]-vvel[2])*(uvel[2]-vvel[2])));
          Re_p = rpart[id*nr+jrho]*rpart[id*nr+jdp]*vel_diff/mu_p;
          M_p = vel_diff/rpart[id*nr+ja];
          rpart[id*nr+jre] = Re_p;
          //compute_re_particles ends

          for (int k =0 ; k < ndim[0] ; k++){
            if (k == 1){
              rpart[id*nr+jfusr+k] = -1;
            }
            else{
              rpart[id*nr+jfusr+k] = 0;
            }
            rdum = 0;
            rdum = rdum+rpart[id*nr+jfusr+k];
            rpart[id*nr+jf0+k] = rdum/pmass;
          }
        }

//---------------------------------------------------------------------------

__global__ void usr_particles_forces_rk3(double *rpart, int n, int nr,int jvol,int jrhop, int jfusr, int jf0,int jrho,int jfqs,int ju0,int jv0, int ja, int jdp, int jre, int jtaup, int jcd){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < n){
//          int i = id/ndim;
//          int j = id%ndim;
          double pmass,pmassf,rdum,cd,S_qs;
          double uvel[3],vvel[3];
          double c0,T0,mu_p,Re_p,M_p,vel_diff;
          pmass = rpart[id*nr+jvol]*rpart[id*nr+jrhop];
          pmassf = rpart[id*nr+jvol]*rpart[id*nr+jrho];
          //compute_re_particles
          uvel[0] = rpart[id*nr+ju0];
          uvel[1] = rpart[id*nr+ju0+1];
          uvel[2] = rpart[id*nr+ju0+2];
          vvel[0] = rpart[id*nr+jv0];
          vvel[1] = rpart[id*nr+jv0+1];
          vvel[2] = rpart[id*nr+jv0+2];
          c0 = 120.0;
          T0 = 291.15;
          c_sound = 1000000; //fix this at the host level so that no concurrency issues arise
          rpart[id*nr+ja] = c_sound;
          mu_p = mu_0;
          vel_diff = sqrt(((uvel[0]-vvel[0])*(uvel[0]-vvel[0]))+((uvel[0]-vvel[0])*(uvel[1]-vvel[1]))+((uvel[2]-vvel[2])*(uvel[2]-vvel[2])));
          Re_p = rpart[id*nr+jrho]*rpart[id*nr+jdp]*vel_diff/mu_p;
          M_p = vel_diff/rpart[id*nr+ja];
          rpart[id*nr+jre] = Re_p;
          //compute_re_particles ends

          for (int k =0 ; k < ndim[0] ; k++){
            if (k == 1){
              rpart[id*nr+jfusr+k] = -1;
            }
            else{
              rpart[id*nr+jfusr+k] = 0;
            }
            //----------usr_particles_f_qs_rk3 starts
            cd = 0;
            S_qs = rpart[id*nr+jvol]*rpart[id*nr+jrhop]/rpart[id*nr+jtaup];
            rpart[id*nr+jcd+k] = cd;
            rpart[id*nr+jfqs+k] = S_qs*(uvel[k] - vvel[k]);
            rdum = 0;
            rdum = rdum+rpart[id*nr+jfusr+k];
            rdum = rdum+rpart[id*nr+jfqs+k];
            rpart[id*nr+jf0+k] = rdum/pmass;
          }
        }

__global__ void update_vel_and_pos_bdf(double *rpart, int n, int ndim, int nr,int jvol,int jrhop, int jfusr, int jf0,int jrho,int jfqs,int ju0,int jv0, int ja, int jdp, int jre, int jtaup, int jcd){
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  double s;
  if(id < n*ndim){
    int i = id/ndim;
    int k = id%ndim;
    // move data to previous position
    rpart[i*nr+ju3+k] = rpart[i*nr+ju2+k];
    rpart[i*nr+ju2+k] = rpart[i*nr+ju1+k];
    rpart[i*nr+ju1+k] = rpart[i*nr+ju0+k];
    rpart[i*nr+jv3+k] = rpart[i*nr+jv2+k];
    rpart[i*nr+jv2+k] = rpart[i*nr+jv1+k];
    rpart[i*nr+jv1+k] = rpart[i*nr+jv0+k];
    rpart[i*nr+jx3+k] = rpart[i*nr+jx2+k];
    rpart[i*nr+jx2+k] = rpart[i*nr+jx1+k];
    rpart[i*nr+jx1+k] = rpart[i*nr+jx0+k];



    // solve for velocity
    s = 1/rpart[i*nr+jtaup];
    rhs = s*((alpha[1]*rpart[i*nr+ju1+k])+(alpha[2]*rpart[i*nr+ju2+k])+(alpha[3]*rpart[i*nr+ju3+k])+rpart[i*nr+jf0+k]+(beta[1]*rpart[i*nr+jv1+k])+(beta[2]*rpart[i*nr+jv2+k])+(beta[3]*rpart[i*nr+jv3+k]);
    rpart[i*nr+jv0+k] = rhs/(beta[0]+s);
    rhx = beta[1]*rpart[i*nr+jx1+k]+beta[2]*rpart[i*nr+jx2+k]+beta[3]*rpart[i*nr+jx3+k]+rpart[i*nr+jv0+k];
    rpart[i*nr+jx0+k] = rhx/beta[0];
  }
}

//----------
__global__ void update_vel_and_pos_rk3(double *rpart, double *kv_stage_p, double *kx_stage_p, int n, int ndim, int nr,int jv0,int jv1,int jv2,int jv3,int ju0,int ju1,int ju2,int ju3, int jx0, int jx1, int jx2, int jx3, int jf0, int fmfac){
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if(id < n*ndim){
      int i = id/ndim;
      int k = id%ndim;
      if (stage == 1){

        // move data to previous position
        rpart[i*nr+ju3+k] = rpart[i*nr+ju2+k];
        rpart[i*nr+ju2+k] = rpart[i*nr+ju1+k];
        rpart[i*nr+ju1+k] = rpart[i*nr+ju0+k];
        rpart[i*nr+jv3+k] = rpart[i*nr+jv2+k];
        rpart[i*nr+jv2+k] = rpart[i*nr+jv1+k];
        rpart[i*nr+jv1+k] = rpart[i*nr+jv0+k];
        rpart[i*nr+jx3+k] = rpart[i*nr+jx2+k];
        rpart[i*nr+jx2+k] = rpart[i*nr+jx1+k];
        rpart[i*nr+jx1+k] = rpart[i*nr+jx0+k];
        kv_stage_p[i*12+k] = rpart[i*nr+jv0+k];
        kx_stage_p[i*12+k] = rpart[i*nr+jx0+k];

      }


      kv_stage_p[i*12+stage*3+k] = rpart[i*nr+jf0+k];
      kx_stage_p[i*12+stage*3+k] = rpart[i*nr+jv0+k];


      if (stage == 3){
        rpart[i*nr+jx0+k] = kx_stage_p[i*12+k]+fmfac*(kx_stage_p[i*12+3+k]+4.0*kx_stage_p[i*12+6+k]+kx_stage_p[i*12+9+k]);
        rpart[i*nr+jv0+k] = kv_stage_p[i*12+k]+fmfac*(kv_stage_p[i*12+3+k]+4.0*kv_stage_p[i*12+6+k]+kv_stage_p[i*12+9+k]);

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

extern "C" void update_particle_location_wrapper_(double *rpart, double *xdrange, int *in_part, int *bc_part, int *n, int *ndim, int *nr, int *jx, int *jx1, int *jx2,int *jx3){

    float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    bool inCPU = false;
    double *d_rpart, *d_xdrange;
    int *d_in_part;*d_bc_part;
//    int ic = 0;
    if(inCPU){

        cudaMalloc(&d_rpart, n[0]*nr[0]*sizeof(double));
        cudaMalloc(&d_xdrange, 6*sizeof(double));
        cudaMalloc(&d_in_part, n[0]*sizeof(int));
        cudaMalloc(&d_bc_part, 6*sizeof(int));
        cudaMemcpy(d_rpart, rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_xerange, xerange, nelt[0]*6*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_in_part, in_part, n[0]*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bc_part, bc_part, 6*sizeof(int), cudaMemcpyHostToDevice);


  }
  else{
      d_rpart= rpart;
      d_xdrange = xdrange;
      d_in_part = in_part;
      d_bc_part = bc_part;
//      cudaMalloc(&d_in_part, n[0]*sizeof(int));
//      cudaMemcpy(d_in_part, in_part, n[0]*sizeof(int), cudaMemcpyHostToDevice);

  }
  int blockSize = 1024, gridSize;
  gridSize = (int)ceil((float)n[0]*ndim[0]/blockSize);
  update_particle_location<<<gridSize, blockSize>>>(d_rpart,d_xdrange,d_inpart,d_bc_part,n[0],ndim[0],nr[0],jx[0],jx1[0],jx2[0],jx3[0]);
//----------need to confirm with Dr. Tania and Dr. Ranka if there is a way to parallelize the memory update - right now im implementing it in the host
//  gridSize = (int)ceil((float)n[0]/blockSize);
//  updated_data_if_outflow<<<gridSize, blockSize>>>(d_rpart,s_ipart,d_inpart,&ic);
  if(inCPU){
      cudaMemcpy(rpart, d_rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(xerange, d_xerange, nelt[0]*6*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(in_part, d_in_part, n[0]*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(bc_part, d_bc_part, 6*sizeof(int), cudaMemcpyDeviceToHost);


      //free
      cudaFree(d_rpart);
      cudaFree(d_bc_part);
      cudaFree(d_xdrange);
      cudaFree(d_in_part);
  }
  else{
//      cudaMemcpy(in_part, d_in_part, n[0]*sizeof(int), cudaMemcpyDeviceToHost);
//        if(nfpts[0]>0){
//            cudaMemcpy(fptsmap, d_fptsmap, nfpts[0]*sizeof(int), cudaMemcpyDeviceToHost);
//            cudaMemcpy(rfpts, d_rfpts, nfpts[0]*nrf[0]*sizeof(double), cudaMemcpyDeviceToHost);
//            cudaMemcpy(ifpts, d_ifpts, nfpts[0]*nif[0]*sizeof(int), cudaMemcpyDeviceToHost);

//        }
      // printf ("print var 1st %d\n", nfpts);
  }
//  cudaFree(d_in_part);
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&time, startEvent, stopEvent);
  // printf ("print var 2nd %d\n", nfpts);
  //printf("particles in nid time is %f\n",time*1e-03);

}

extern "C" void interp_props_part_location_wrapper_(double *rpart,int *ipart,double *vx, double *vy, double *vz,double *t, double *vtrans, double *rep, double *xgll, double *ygll, double *zgll, double *wxgll, double *wygll, double *wzgll, int* n, int* nr, int* ni, int* nx1, int* nx1r, int* jr,int* ju0,int* je0,int* jtemp,int* jrho,int* nelt){

    float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    bool inCPU = false;
    double *d_rpart,*d_vx,*d_vy, *d_vz, *d_t, *d_vtrans;
    int *d_ipart;
    if(inCPU){

        cudaMalloc(&d_rpart, n[0]*nr[0]*sizeof(double));
        cudaMalloc(&d_ipart, n[0]*ni[0]*sizeof(int));
        cudaMalloc(&d_vx, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double));
        cudaMalloc(&d_vy, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double));
        cudaMalloc(&d_vz, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double));
        cudaMalloc(&d_t, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double));
        cudaMalloc(&d_vtrans, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double));
       cudaMalloc(&d_rep, nx1[0]*nx1[0]*nx1[0]*sizeof(double));
       cudaMalloc(&d_xgll, nx1[0]*sizeof(double));
       cudaMalloc(&d_ygll, nx1[0]*sizeof(double));
       cudaMalloc(&d_zgll, nx1[0]*sizeof(double));
       cudaMalloc(&d_wxgll, nx1[0]*sizeof(double));
       cudaMalloc(&d_wygll, nx1[0]*sizeof(double));
       cudaMalloc(&d_wzgll, nx1[0]*sizeof(double));
       cudaMemcpy(d_rpart, rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_ipart, ipart, n[0]*ni[0]*sizeof(int), cudaMemcpyHostToDevice);
       cudaMemcpy(d_vx, vx, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_vy, vy, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_vz, vz, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_t, t, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_vtrans, vtrans, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_rep, rep, nx1[0]*nx1[0]*nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_xgll, xgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_ygll, ygll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_zgll, zgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_wxgll, wxgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_wygll, wygll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(d_wzgll, wzgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);


  }
  else{
      d_rpart= rpart;
      d_ipart = ipart;
      d_vx = vx;
      d_vy = vy;
      d_vz = vz;
      d_t =t;
      d_vtrans = vtrans;
      cudaMalloc(&d_rep, nx1[0]*nx1[0]*nx1[0]*sizeof(double));
      cudaMalloc(&d_xgll, nx1[0]*sizeof(double));
      cudaMalloc(&d_ygll, nx1[0]*sizeof(double));
      cudaMalloc(&d_zgll, nx1[0]*sizeof(double));
      cudaMalloc(&d_wxgll, nx1[0]*sizeof(double));
      cudaMalloc(&d_wygll, nx1[0]*sizeof(double));
      cudaMalloc(&d_wzgll, nx1[0]*sizeof(double));
      cudaMemcpy(d_rep, rep, nx1[0]*nx1[0]*nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_xgll, xgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ygll, ygll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_zgll, zgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_wxgll, wxgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_wygll, wygll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_wzgll, wzgll, nx1[0]*sizeof(double), cudaMemcpyHostToDevice);
  }
  int blockSize = 1024, gridSize;
  gridSize = (int)ceil((float)n[0]/blockSize);
  interp_props_part_location<<<gridSize, blockSize>>>(d_rpart,d_vx,d_vy,d_vz,d_t,d_vtrans,d_rep,d_xgll,d_ygll,d_zgll,d_wxgll,d_wygll,d_wzgll,n[0],nr[0],ni[0],nx1[0],nx1r[0],jr[0],ju0[0],je0[0],jtemp[0],jrho[0]);
  if(inCPU){
      cudaMemcpy(ipart, d_ipart, n[0]*ni[0]*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(rpart, d_rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(vx, d_vx, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(vy, d_vy, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(vz, d_vz, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(t, d_t, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(vtrans, d_vtrans, nx1[0]*nx1[0]*nx1[0]*nelt[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(rep, d_rep, nx1[0]*nx1[0]*nx1[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(xgll, d_xgll, nx1[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(ygll, d_ygll, nx1[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(zgll, d_zgll, nx1[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(wxgll, d_wxgll, nx1[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(wygll, d_wygll, nx1[0]*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(wzgll, d_wzgll, nx1[0]*sizeof(double), cudaMemcpyDeviceToHost);

      //free
      cudaFree(d_rpart);
      cudaFree(d_ipart);
      cudaFree(d_vx);
      cudaFree(d_vy);
      cudaFree(d_vz);
      cudaFree(d_t);
      cudaFree(d_vtrans);
     cudaFree(d_rep);
     cudaFree(d_xgll);
     cudaFree(d_ygll);
     cudaFree(d_zgll);
     cudaFree(d_wxgll);
     cudaFree(d_wygll);
     cudaFree(d_wzgll);
  }

  else{
       cudaMemcpy(rep,d_rep, nx1[0]*nx1[0]*nx1[0]*sizeof(double), cudaMemcpyDeviceToHost);

        //free
       cudaFree(d_rep);
       cudaFree(d_xgll);
       cudaFree(d_ygll);
       cudaFree(d_zgll);
       cudaFree(d_wxgll);
       cudaFree(d_wygll);
       cudaFree(d_wzgll);

 }
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&time, startEvent, stopEvent);

  }

 //---------------------------------------------------------------------------
 extern "C" void usr_particles_forces_bdf_wrapper_(double *rpart, int *n, int* nr,int* jvol,int* jrhop, int* jfusr, int* jf0,int* jrho,int* jfqs,int* ju0,int* jv0, int* ja, int* jdp, int* jre, int* jtaup, int* jcd){

     float time;
     cudaEvent_t startEvent, stopEvent;
     cudaEventCreate(&startEvent);
     cudaEventCreate(&stopEvent);
     cudaEventRecord(startEvent, 0);

     bool inCPU = false;
     double *d_rpart;
     if(inCPU){
       cudaMalloc(&d_rpart, n[0]*nr[0]*sizeof(double));
       cudaMemcpy(d_rpart, rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyHostToDevice);
     }
     else{
       d_rpart= rpart;
     }
     int blockSize = 1024, gridSize;
     gridSize = (int)ceil((float)n[0]/blockSize);
     usr_particles_forces_bdf<<<gridSize, blockSize>>>(d_rpart,n[0],nr[0],jvol[0],jrhop[0],jfusr[0],jf0[0],jrho[0],jfqs[0],ju0[0],jv0[0],ja[0],jdp[0],jre[0],jtaup[0],jcd[0]);
     if(inCPU){
         cudaMemcpy(rpart, d_rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyDeviceToHost);
         //free
         cudaFree(d_rpart);
     }

     cudaEventRecord(stopEvent, 0);
     cudaEventSynchronize(stopEvent);
     cudaEventElapsedTime(&time, startEvent, stopEvent);
}


//---------------------------------------------------------

//---------------------------------------------------------------------------
extern "C" void usr_particles_forces_rk3_wrapper_(double *rpart, int* n, int* nr, int* jvol,int* jrhop, int* jfusr, int* jf0,int* jrho,int* jfqs,int* ju0,int* jv0, int* ja, int* jdp, int* jre, int* jtaup, int* jcd){

    float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    bool inCPU = false;
    double *d_rpart;
    if(inCPU){
      cudaMalloc(&d_rpart, n[0]*nr[0]*sizeof(double));
      cudaMemcpy(d_rpart, rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyHostToDevice);
    }
    else{
      d_rpart= rpart;
    }
    int blockSize = 1024, gridSize;
    gridSize = (int)ceil((float)n[0]/blockSize);
    usr_particles_forces_rk3<<<gridSize, blockSize>>>(d_rpart,n[0],nr[0],jvol[0],jrhop[0],jfusr[0],jf0[0],jrho[0],jfqs[0],ju0[0],jv0[0],ja[0],jdp[0],jre[0],jtaup[0],jcd[0]);
    if(inCPU){
        cudaMemcpy(rpart, d_rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyDeviceToHost);
        //free
        cudaFree(d_rpart);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
}


//---------------------------------------------------------
extern "C" void update_vel_and_pos_bdf_wrapper_(double *rpart, int *n, int* nr,int* jvol,int* jrhop, int* jfusr, int* jf0,int* jrho,int* jfqs,int* ju0,int* jv0, int* ja, int* jdp, int* jre, int* jtaup, int* jcd){

    float time;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    bool inCPU = false;
    double *d_rpart;
    if(inCPU){
      cudaMalloc(&d_rpart, n[0]*nr[0]*sizeof(double));
      cudaMemcpy(d_rpart, rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyHostToDevice);
    }
    else{
      d_rpart= rpart;
    }
    int blockSize = 1024, gridSize;
    gridSize = (int)ceil((float)n[0]/blockSize);
    update_vel_and_pos_bdf<<<gridSize, blockSize>>>
    if(inCPU){
        cudaMemcpy(rpart, d_rpart, n[0]*nr[0]*sizeof(double), cudaMemcpyDeviceToHost);
        //free
        cudaFree(d_rpart);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
}
