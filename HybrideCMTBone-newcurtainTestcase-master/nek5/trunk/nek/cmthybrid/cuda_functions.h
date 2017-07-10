// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//program
double get_time( void );
int test_( int matsize, unsigned int gridsize, int jobs, double* h_A, 
          double* h_AA, int M, int N, int K);
void init_matrix(double * mat, int size, int begin);
void get_transpose(double *t, double * u, int N);
// void full2facewrapper_(double *vols, double*faces, int nel, int n, int nxyz, int*iface, bool device_arr, bool pull_result);
