__global__ void kernel(int *buf, int len) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int leap = blockDim.x * gridDim.x;

    for (int i = ix; i < len; i += leap) {
        buf[i] = ix;
    }
}
