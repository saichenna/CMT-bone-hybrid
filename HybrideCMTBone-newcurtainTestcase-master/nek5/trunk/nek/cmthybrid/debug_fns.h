//Debug functions
#include <sys/time.h>
void print(double * mat, int M, int N){
	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++){
			printf("%f, ", mat[i+j*M]);
		}
		printf("\n");
	}
	printf("\n");

}

void print_wolfram(double * mat, int M, int N){
	printf("{");
	for(int i=0; i<M; i++){
		printf("{");
		for(int j=0; j<N; j++){
			printf("%f", mat[i+j*M]);
			if(j<N-1) printf(",");
		}
		printf("}");
		if(i<M-1) printf(",");
	}
	printf("}\n");
	printf("\n");
}
