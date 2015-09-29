__global__ void multiplyBy2(int *in, int *out, long size) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < size) {
        out[ix] = in[ix] * 2;
    }
}

__global__ void sum(int *input, int *output, long size, int stage, int totalStages) {
    const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
    const int jump = 4 * 32;
    if (stage == 0) {
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
