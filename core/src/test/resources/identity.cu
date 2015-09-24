__global__ void identity(long size, int *input, int *output) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    if (ix < size) {
        output[ix] = input[ix];
    }
}
