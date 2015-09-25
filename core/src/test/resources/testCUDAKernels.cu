// very simple test kernel
__global__ void identity(const int *input, int *output, long size) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        output[ix] = input[ix];
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
