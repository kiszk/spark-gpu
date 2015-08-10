__global__ void sum(int *x, size_t n) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int leap = blockDim.x * gridDim.x;

    int res = 0;

    {
#define UNROLL_AMOUNT 16
        const size_t unrollN = n / (UNROLL_AMOUNT * leap) * (UNROLL_AMOUNT * leap);
        for (size_t i = ix; i < unrollN;) {
#pragma unroll 16
            for (int j = 0; j < UNROLL_AMOUNT; ++j, i += leap) {
                res += x[i];
            }
        }
        for (size_t i = unrollN + ix; i < n; i += leap) {
            res += x[i];
        }
    }

    x[ix] = res;
}
