#include<math.h>

#define	GET_BLOB_ADDRESS(ptr, offset)	(&((ptr)[(offset)/sizeof((ptr)[0])]))
#define	GET_ARRAY_CAPACITY(ptr)		(((long *)(ptr))[0])
#define	GET_ARRAY_LENGTH(ptr)		(((long *)(ptr))[1])
#define	GET_ARRAY_BODY(ptr)		(&((ptr)[128/sizeof((ptr)[0])]))
#define	SET_ARRAY_CAPACITY(ptr, val)	{ (((long *)(ptr))[0]) = (val); }
#define	SET_ARRAY_LENGTH(ptr, val)	{ (((long *)(ptr))[1]) = (val); }

// very simple test kernel
__global__ void identity(const int *input, int *output, long size) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        output[ix] = input[ix];
    }
}

// very simple test kernel for int array
__global__ void intArrayIdentity(const long *input, const int *inputBlob, long *output, int *outputBlob, long size) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        // copy int array
        long offset = input[ix];
        const int *inArray = GET_BLOB_ADDRESS(inputBlob, offset);
        const long capacity = GET_ARRAY_CAPACITY(inArray);
        const long length   = GET_ARRAY_LENGTH(inArray);
        const int *inArrayBody = GET_ARRAY_BODY(inArray);

        int *outArray = GET_BLOB_ADDRESS(outputBlob, offset);
        int *outArrayBody = GET_ARRAY_BODY(outArray);
        for (long i = 0; i < length; i++) {
          outArrayBody[i] = inArrayBody[i];
        }
        output[ix] = offset;
        SET_ARRAY_CAPACITY(outArray, capacity);
        SET_ARRAY_LENGTH(outArray, length);
    }
}

// very simple test kernel for IntDataPoint class
__global__ void IntDataPointIdentity(const long *inputX, const int *inputY, const int *inputBlob, long *outputX, int *outputY, int *outputBlob, long size) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        // copy int array
        long offset = inputX[ix];
        const int *inArray = GET_BLOB_ADDRESS(inputBlob, offset);
        const long capacity = GET_ARRAY_CAPACITY(inArray);
        const long length   = GET_ARRAY_LENGTH(inArray);
        const int *inArrayBody = GET_ARRAY_BODY(inArray);

        int *outArray = GET_BLOB_ADDRESS(outputBlob, offset);
        int *outArrayBody = GET_ARRAY_BODY(outArray);
        for (long i = 0; i < length; i++) {
          outArrayBody[i] = inArrayBody[i];
        }
        outputX[ix] = offset;
        SET_ARRAY_CAPACITY(outArray, capacity);
        SET_ARRAY_LENGTH(outArray, length);

        // copy int scalar value
        outputY[ix] = inputY[ix];
    }
}

// very simple test kernel for int array with free var
__global__ void intArrayAdd(const long *input, const int *inputBlob, long *output, int *outputBlob, long size, const int *inFreeArray) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        // copy int array
        long offset = input[ix];
        const int *inArray = GET_BLOB_ADDRESS(inputBlob, offset);
        const long capacity = GET_ARRAY_CAPACITY(inArray);
        const long length   = GET_ARRAY_LENGTH(inArray);
        const int *inArrayBody = GET_ARRAY_BODY(inArray);

        int *outArray = GET_BLOB_ADDRESS(outputBlob, offset);
        int *outArrayBody = GET_ARRAY_BODY(outArray);
        for (long i = 0; i < length; i++) {
          outArrayBody[i] = inArrayBody[i] + inFreeArray[i];
        }
        output[ix] = offset;
        SET_ARRAY_CAPACITY(outArray, capacity);
        SET_ARRAY_LENGTH(outArray, length);
    }
}

// test kernel for multiple input columns
__global__ void vectorLength(const double *x, const double *y, double *len, long size) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        len[ix] = sqrt(x[ix] * x[ix] + y[ix] * y[ix]);
    }
}

// test kernel for multiple input and multiple output columns, with different types
__global__ void plusMinus(const double *base, const float *deviation, double *a, float *b, long size) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        a[ix] = base[ix] - deviation[ix];
        b[ix] = base[ix] + deviation[ix];
    }
}

// test kernel for two const arguments
__global__ void applyLinearFunction(const short *x, short *y, long size, short a, short b) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        y[ix] = a + b * x[ix];
    }
}

// test kernel for custom number of blocks + const argument
// manual SIMD, to be ran on size / 8 threads, assumes size % 8 == 0
// note that key is reversed, since it's little endian
__global__ void blockXOR(const char *input, char *output, long size, long key) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix * 8 < size) {
        ((long *)output)[ix] = ((const long *)input)[ix] ^ key;
    }
}

// another simple test kernel
__global__ void multiplyBy2(int *in, int *out, long size) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < size) {
        out[ix] = in[ix] * 2;
    }
}

// test reduce kernel that sums elements
__global__ void sum(int *input, int *output, long size, int stage, int totalStages) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    const int jump = 64 * 256;
    if (stage == 0) {
        if (ix < size) {
            assert(jump == blockDim.x * gridDim.x);
            int result = 0;
            for (long i = ix; i < size; i += jump) {
                result += input[i];
            }
            input[ix] = result;
        } 
    } else if (ix == 0) {
        const long count = (size < (long)jump) ? size : (long)jump;
        int result = 0;
        for (long i = 0; i < count; ++i) {
            result += input[i];
        }
        output[0] = result;
    }
}

// test reduce kernel that sums elements
__global__ void intArraySum(const long *input, const int *inputBlob, long *output, int *outputBlob, long size, int stage, int totalStages) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    const int jump = 64 * 256;
    if (stage == 0) {
        if (ix < size) {
            assert(jump == blockDim.x * gridDim.x);
            const int *accArray = GET_BLOB_ADDRESS(inputBlob, input[ix]);
            int *accArrayBody = const_cast<int *>GET_ARRAY_BODY(accArray);
            for (long i = ix + jump; i < size; i += jump) {
                long offset = input[i];
                const int *inArray = GET_BLOB_ADDRESS(inputBlob, offset);
                const long length = GET_ARRAY_LENGTH(inArray);
                const int *inArrayBody = GET_ARRAY_BODY(inArray);

                for (long j = 0; j < length; j++) {
                     accArrayBody[j] += inArrayBody[j];
                }
            }
        }
    } else if (ix == 0) {
        const long count = (size < (long)jump) ? size : (long)jump;
        int *outArray = GET_BLOB_ADDRESS(outputBlob, input[ix]);
        int *outArrayBody = GET_ARRAY_BODY(outArray);
        long capacity = 0, length = 0;
        for (long i = 0; i < count; i++) { 
            const long offset = input[i];
            const int *inArray = GET_BLOB_ADDRESS(inputBlob, offset);
            capacity = GET_ARRAY_CAPACITY(inArray);
            length   = GET_ARRAY_LENGTH(inArray);
            const int *inArrayBody = GET_ARRAY_BODY(inArray);

            if (i == 0) {
                for (long j = 0; j < length; j++) {
                    outArrayBody[j] = 0;
                }
            } 
            for (long j = 0; j < length; j++) {
                outArrayBody[j] += inArrayBody[j];
            }
        } 
        output[ix] = 0;
        SET_ARRAY_CAPACITY(outArray, capacity);
        SET_ARRAY_LENGTH(outArray, length);
   } 
}

// map for DataPoint class
__global__ void DataPointMap(const long *inputX, const int *inputY, const double *inputBlob, long *output, double *outputBlob, long size, const double *inFreeArray) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        // copy int array
        long offset = inputX[ix];
        const double *inArray = GET_BLOB_ADDRESS(inputBlob, offset);
        const long capacity = GET_ARRAY_CAPACITY(inArray);
        const long length   = GET_ARRAY_LENGTH(inArray);
        const double *inArrayBody = GET_ARRAY_BODY(inArray);

        double *outArray = GET_BLOB_ADDRESS(outputBlob, offset);
        double *outArrayBody = GET_ARRAY_BODY(outArray);
        for (long i = 0; i < length; i++) {
          outArrayBody[i] = inArrayBody[i] + inFreeArray[i];
        }
        output[ix] = offset;
        SET_ARRAY_CAPACITY(outArray, capacity);
        SET_ARRAY_LENGTH(outArray, length);
    }
}

// reduce for DataPoint class
__global__ void DataPointReduce(const long *input, const double *inputBlob, long *output, double *outputBlob, long size, int stage, int totalStages) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    const int jump = 64 * 256;
    if (stage == 0) {
        if (ix < size) {
            assert(jump == blockDim.x * gridDim.x);
            const double *accArray = GET_BLOB_ADDRESS(inputBlob, input[ix]);
            double *accArrayBody = const_cast<double *>GET_ARRAY_BODY(accArray);
            for (long i = ix + jump; i < size; i += jump) {
                long offset = input[i];
                const double *inArray = GET_BLOB_ADDRESS(inputBlob, offset);
                const long length   = GET_ARRAY_LENGTH(inArray);
                const double *inArrayBody = GET_ARRAY_BODY(inArray);

                for (long j = 0; j < length; j++) {
                     accArrayBody[j] += inArrayBody[j];
                }
            }
        }
    } else if (ix == 0) {
        const long count = (size < (long)jump) ? size : (long)jump;
        double *outArray = GET_BLOB_ADDRESS(outputBlob, input[ix]);
        double *outArrayBody = GET_ARRAY_BODY(outArray);
        long capacity = 0, length = 0;
        for (long i = 0; i < count; i++) { 
            const long offset = input[i];
            const double *inArray = GET_BLOB_ADDRESS(inputBlob, offset);
            capacity = GET_ARRAY_CAPACITY(inArray);
            length   = GET_ARRAY_LENGTH(inArray);
            const double *inArrayBody = GET_ARRAY_BODY(inArray);

            if (i == 0) {
                for (long j = 0; j < length; j++) {
                    outArrayBody[j] = 0;
                }
            } 
            for (long j = 0; j < length; j++) {
                outArrayBody[j] += inArrayBody[j];
            }
        } 
        output[ix] = 0;
        SET_ARRAY_CAPACITY(outArray, capacity);
        SET_ARRAY_LENGTH(outArray, length);
   } 
}

// map for Logistic regression
__device__ double sdotvv(const double * __restrict__ x, const double * __restrict__ y, int n) {
    double ans = 0.0;
    for(int i = 0; i < n; i++) {
        ans += x[i] * y[i];
    }
    return ans;
}
__device__ void dmulvs(double *result, const double * __restrict__ x, double c, int n) {
    for(int i = 0; i < n; i++) {
        result[i] = x[i] * c;
    }
}
__device__ void map(double *result, const double * __restrict__ x, double y, const double * __restrict__ w, int n) {
    dmulvs(result, x, (1 / (1 + exp(-y * (sdotvv(w, x, n)))) - 1) * y, n);
}

__global__ void LRMap(const long * __restrict__ inputX, const double *  __restrict__ inputY, const double * __restrict__ inputBlob, long *output, double *outputBlob, long size, const double * __restrict__ inputW) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        // copy int array
        long offset = inputX[ix];
        const double *inArray = GET_BLOB_ADDRESS(inputBlob, offset);
        const long capacity = GET_ARRAY_CAPACITY(inArray);
        const long length   = GET_ARRAY_LENGTH(inArray);
        const double * inArrayBody = GET_ARRAY_BODY(inArray);

        double *outArray = GET_BLOB_ADDRESS(outputBlob, offset);
        double *outArrayBody = GET_ARRAY_BODY(outArray);

        map(outArrayBody, inArrayBody, inputY[ix], inputW, length);

        output[ix] = offset;
        SET_ARRAY_CAPACITY(outArray, capacity);
        SET_ARRAY_LENGTH(outArray, length);
    }
}

__device__ inline double *  __shfl_down_double(double *var, unsigned int srcLane, int width=32) {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return *reinterpret_cast<double**>(&a);
}

__global__ void LRReduce(const long * __restrict__ input, const double * __restrict__ inputBlob, long *output, double *outputBlob, long size/*, int stage, int totalStages*/) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx == 0) {
        double *outArray = GET_BLOB_ADDRESS(outputBlob, input[idx]);
        double *outArrayBody = GET_ARRAY_BODY(outArray);
 
        long capacity = 0, length = 0;
        for (long i = 0; i < size; i++) {
            long offset = input[i];
            const double *inArray = GET_BLOB_ADDRESS(inputBlob, offset);
            capacity = GET_ARRAY_CAPACITY(inArray);
            length   = GET_ARRAY_LENGTH(inArray);
            const double * __restrict__ inArrayBody = GET_ARRAY_BODY(inArray);

            if (i == 0) {
                for (long j = 0; j < length; j++) {
                    outArrayBody[j] = 0;
                }
            } 

            for (long j = 0; j < length; j++) {
                outArrayBody[j] += inArrayBody[j];
            }
        }
        output[idx] = 0;
        SET_ARRAY_CAPACITY(outArray, capacity);
        SET_ARRAY_LENGTH(outArray, length);
    }

#if 0
    int warpCnt = (threadsPerBlock + warpSize-1)/warpSize;
    int laneId  = threadIdx.x % warpSize;
    int warpId  = threadIdx.x / warpSize;
    double *val = NULL;
    double *val1 = NULL;
    // shared memory is allocated per block, not per thread. All threads/warps inside a block share a single instance of variable. 
    static __shared__ double warpResults[32][D]; // The maximum number of possible warps inside a block is 32 (since max thrdsPerBlock=1024 i.e 32 * 32)

    // All threads inside warp execute synchronously
    if (idx < size) {
        const double *accArray = GET_BLOB_ADDRESS(inputBlob, input[idx]);
        double *accArrayBody = const_cast<double *>GET_ARRAY_BODY(accArray);

        /* FOLD EACH WARP(Set of 32 threds/entries)
            For each warpSet(32 entries),do the following steps
            1) add last 16 entries into first 16 
                If total launched thread is 17, then it would be like 0 + 16, 1+0, 2+0 .. size
                shufl_down would return zero when there is no thread.
            2) add second 8 entries into first 8 entries
            3) and so on... till you add second entry in first entry
            4) The first thread's val variable in each  warp will contain the sum of all 32 val variable 
        */

        for (int foldSize = warpSize/2; foldSize; foldSize/=2) {
            val1 = __shfl_down_double(val, foldSize);
            if (val1 != NULL)
                for(int i = 0; i < D; i++) val[i] += val1[i];
    }

    // Lets sequentially store each warp's sum into array 'a' 
    if (laneId == 0) {
        memcpy(warpResults[warpId], val, sizeof(*val) * D);
    }
}
    // Wait for all warps inside a block to complete.
    __syncthreads();

    if (idx == 0) {
        for (int w = 0; w < warpCnt;w++)
            for (int i = 0; i < D;i++)
                result[i] += warpResults[w][i];
    }

   //Let each block copy the results into temp result array.
#endif
}