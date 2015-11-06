#include<math.h>
#include<stdlib.h>
#include<stdio.h>

#define D 5
#define N 2
#define ITERATIONS 10
#define thPerBlock  N
#define noOfBlocks ((N + thPerBlock - 1)/thPerBlock)


typedef struct DataPoint {
	double x[D];
 	double y;
}DataPoint;

__device__
double dot(double x[D], double y[D]) {
    double ans = 0.0;
    int i = 0;
    while( i < D) {
        ans += x[i] * y[i];
        i += 1;
    }
    return ans;
}

__device__ void
muls(double result[D], double x[D], double c) {
    int i=0;
    for(i=0; i<D; i++) {
        result[i] = x[i] * c;
    }
}

__device__ void 
map(double result[D], double x[D], double y, double w[D]) {
    muls(result, x, (1 / (1 + exp(-y * (dot(w, x)))) - 1) * y);
}

__device__ void 
add(double result[D], double x[D], double y[D]) {
    int i=0;
    for(i=0; i<D; i++) {
        result[i] = x[i] * y[i];
    }
}

void
subself(double x[D], double y[D]) {
    int i;
    for(i=0; i<D; i++)
        x[i] -= y[i];
}

__global__ void
mapAll(int count, double result[N * D], double x[N * D], double y[D], double w[D]) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < count)
    map(&result[idx * D], &x[idx * D ], y[idx],w);
}

__global__ void
reduceAll(int count, double result[D], double inp[N * D]) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int j;
    //if(idx < count)
    for(idx=0;idx<2;idx++)
    for(j=0;j<D;j++)
    {
        result[j] += inp[idx * D + j];
    } 
}

void
print(char a[], double p[D]) {
    int i=0;
    printf("%s",a);
    for(i=0;i<D;i++)
        printf("%e, ",p[i]);
    printf("\n");
}

int
main(int argc, char * argv[])
{
    //DataPoint DataSet[N] = { {{1.0,2.0,3.0,4.0,5.0},1.0 }, {{1.0,2.0,3.0,4.0,5.0},1.0 } };
    double x[N][D] = { {1.0,2.0,3.0,4.0,5.0}, {1.0,2.0,3.0,4.0,5.0}};
    double *x1 = (double *) x;
    double y[D]    = { 1.0, 1.0 };
    double w[D] = { 0.9706523792091876, 0.9689950601959192, 0.8871562128145087, 0.64360369303987, 0.905075008043427 };
    double mr[N][D],rr[D];
    double *mr1 = (double *) mr;
    int i=0, cnt,*cnt_d;
    double *mr1_d, *x1_d, *y_d,*w_d, *rr_d;

    cudaMalloc( (void **) &mr1_d, N * D * sizeof(double));
    cudaMalloc( (void **) &rr_d, D * sizeof(double));

    cudaMalloc( (void **) &x1_d,  N * D * sizeof(double));
    cudaMemcpy( x1_d, x1, N * D * sizeof(double) , cudaMemcpyHostToDevice );

    cudaMalloc( (void **) &y_d, N * sizeof(double));
    cudaMemcpy( y_d, y, N * sizeof(double) , cudaMemcpyHostToDevice );


    cudaMalloc( (void **) &w_d, D * sizeof(double));
    cudaMemcpy( w_d, w, D * sizeof(double) , cudaMemcpyHostToDevice );

    cudaMalloc( (void **) &cnt_d, sizeof(int));
    cnt = N;
    cudaMemcpy( cnt_d, &cnt, sizeof(int) , cudaMemcpyHostToDevice );

    for(i=0;i<ITERATIONS;i++) {
        double rr[D] = { 0 };

        cudaMemcpy( w_d, w, D * sizeof(double) , cudaMemcpyHostToDevice );
        cudaMemcpy( rr_d, rr, D * sizeof(double) , cudaMemcpyHostToDevice );

        // TODO - findout how you were able to pass cnt - may be stack is passed??
        mapAll<<<noOfBlocks,thPerBlock>>>(cnt,mr1_d,x1_d,y_d,w_d); 
        cudaMemcpy(mr1, mr1_d, N * D * sizeof(double) , cudaMemcpyDeviceToHost );
        //print("MAP 0 ", mr1);
        //print("MAP 1 ", mr1 + D);

        // TODO - need to find a way to do cuda thread syncs
        // This reduce is in GPU is useless
        reduceAll<<<1,1>>>(cnt,rr_d,mr1_d);
        cudaMemcpy(rr, rr_d, D * sizeof(double) , cudaMemcpyDeviceToHost );
        //print("rr--->", rr);
        //print("W--->", w);

        subself(w,rr);
        //print("(W(-)>", w);
        //printf("\n\n");

    }
    cudaMemcpy(rr, rr_d, D * sizeof(double) , cudaMemcpyDeviceToHost );
    print("rr--->", rr);


    cudaFree(mr1_d);
    cudaFree(cnt_d);
    cudaFree(x1_d);
    cudaFree(y_d);
    cudaFree(w_d);
    cudaFree(rr_d);


    return 0;

}
