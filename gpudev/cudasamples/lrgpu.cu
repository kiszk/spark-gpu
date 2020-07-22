#include<math.h>
#include<stdlib.h>
#include<stdio.h>

#define D 5
#define N 2
#define ITERATIONS 10
#define threadsPerBlock blockDim.x
#define noOfBlocks      gridDim.x
#include "lrgpu.h"


__global__
void blockReduce(int count, double * result, double *data) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpCnt = (threadsPerBlock + warpSize-1)/warpSize;
    int laneId  = threadIdx.x % warpSize;
    int warpId  = threadIdx.x / warpSize;
    double *val = NULL;
    double *val1 = NULL;
    // shared memory is allocated per block, not per thread. All threads/warps inside a block share a single instance of variable. 
    static __shared__ double warpResults[32][D]; // The maximum number of possible warps inside a block is 32 (since max thrdsPerBlock=1024 i.e 32 * 32)

    // All threads inside warp execute synchronously
    if(idx < count)
        val = data + (idx * D);

    /* FOLD EACH WARP(Set of 32 threds/entries)
        For each warpSet(32 entries),do the following steps
        1) add last 16 entries into first 16 
            If total launched thread is 17, then it would be like 0 + 16, 1+0, 2+0 .. size 
            shufl_down would return zero when there is no thread.
        2) add second 8 entries into first 8 entries
        3) and so on... till you add second entry in first entry
        4) The first thread's val variable in each  warp will contain the sum of all 32 val variable 
    */
    

    for(int foldSize=warpSize/2; foldSize; foldSize/=2) {
            val1 = __shfl_down_double(val, foldSize);
            if(val1 != NULL)
               for(int i=0;i<D;i++) val[i] += val1[i];
    }

    // Lets sequentially store each warp's sum into array 'a' 
    if(laneId==0) {
        memcpy(warpResults[warpId],val,sizeof(*val) * D);
    }

    // Wait for all warps inside a block to complete.
    __syncthreads();

    if(idx==0) {
        for(int w=0; w<warpCnt;w++)
            for(int i=0; i<D;i++)
                result[i]+=warpResults[w][i];
    }

   //Let each block copy the results into temp result array.
} 

__device__ void 
map(double result[D], double x[D], double y, double w[D]) {
    muls(result, x, (1 / (1 + exp(-y * (dot(w, x)))) - 1) * y);
}

__global__ void
mapAll(int count, double result[N * D], double x[N * D], double y[D], double w[D]) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < count)
        map(&result[idx * D], &x[idx * D ], y[idx],w);
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

        mapAll<<<1,N>>>(cnt,mr1_d,x1_d,y_d,w_d); 
        cudaMemcpy(mr1, mr1_d, N * D * sizeof(double) , cudaMemcpyDeviceToHost );
        //print("MAP 0 ", mr1);
        //print("MAP 1 ", mr1 + D);

        blockReduce<<<1,N>>>(cnt,rr_d,mr1_d);
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
