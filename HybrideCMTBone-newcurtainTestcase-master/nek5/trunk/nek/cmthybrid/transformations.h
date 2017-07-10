//Functions for transformation/access
void get_s_mat(double * result, double * s, int M, int ind){
	for(int i = 0, ii = 0; i<M ; i++, ii++){
		for(int j = M*ind, jj=0 ; j<M*(ind+1); j++, jj++){
			result[ii+M*jj] = s[i+j*M];
		}
	}
}

void get_t_mat(double *t, double* u, int M, int N){
	for(int k=0; k<N; k++){
		for(int j=0, jj=0; j<M*M; j++, jj+=M){
			if(jj>=M*M) jj=jj%M + 1;
			for(int i=0; i<M; i++){
				t[k*M*M*M+i+j*M] = u[k*M*M*M+i+jj*M];
			}
		}
	}
}

void untransform_t(double *t, double *u, int M, int N){
	for(int k=0; k<N; k++){
		for(int j=0, jj=0; j<M*M; j++, jj+=M){
			if(jj>=M*M) jj=jj%M + 1;
			for(int i=0; i<M; i++){
				t[k*M*M*M+i+jj*M] = u[k*M*M*M+i+j*M];	
			}
		}	
	}
}

