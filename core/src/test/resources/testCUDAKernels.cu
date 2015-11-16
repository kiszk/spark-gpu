#define	GET_BLOB_ADDRESS(ptr, offset)	(&(((char *)(ptr))[(offset)]))
#define	GET_ARRAY_CAPACITY(ptr)		(((long *)(ptr))[0])
#define	GET_ARRAY_LENGTH(ptr)		(((long *)(ptr))[1])
#define	GET_ARRAY_BODY(ptr)		(&(((char *)(ptr))[128]))
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
__global__ void intArrayIdentity(const long *input, const char *inputBlob, long *output, char *outputBlob, long size) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        // copy int array
        long offset = input[ix];
        const char *inArray = GET_BLOB_ADDRESS(inputBlob, offset);
        const long capacity = GET_ARRAY_CAPACITY(inArray);
        const long length   = GET_ARRAY_LENGTH(inArray);
        const int *inArrayBody = (int *)GET_ARRAY_BODY(inArray);

        char *outArray = GET_BLOB_ADDRESS(outputBlob, offset);
        int *outArrayBody = (int *)GET_ARRAY_BODY(outArray);
        for (long i = 0; i < length; i++) {
          outArrayBody[i] = inArrayBody[i];
        }
        output[ix] = offset;
        SET_ARRAY_CAPACITY(outArray, capacity);
        SET_ARRAY_LENGTH(outArray, length);
    }
}

// very simple test kernel for IntDataPoint class
__global__ void IntDataPointIdentity(const long *inputX, const int *inputY, const char *inputBlob, long *outputX, int *outputY, char *outputBlob, long size) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        // copy int array
        long offset = inputX[ix];
        const char *inArray = GET_BLOB_ADDRESS(inputBlob, offset);
        const long capacity = GET_ARRAY_CAPACITY(inArray);
        const long length   = GET_ARRAY_LENGTH(inArray);
        const int *inArrayBody = (int *)GET_ARRAY_BODY(inArray);

        char *outArray = GET_BLOB_ADDRESS(outputBlob, offset);
        int *outArrayBody = (int *)GET_ARRAY_BODY(outArray);
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
        assert(jump == blockDim.x * gridDim.x);
        int result = 0;
        for (long i = ix; i < size; i += jump) {
            result += input[i];
        }
        input[ix] = result;
    } else if (ix == 0) {
        int result = 0;
        for (long i = 0; i < jump; ++i) {
            result += input[i];
        }
        output[0] = result;
    }
}
