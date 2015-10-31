#include <curand_kernel.h>

__global__ void SparkGPUPi_map(const int *input, int *output, long size) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  curandState s;
  int seed = 1;
  curand_init(seed, idx, 0, &s);

  float x = curand_uniform(&s) * 2 - 1;
  float y = curand_uniform(&s) * 2 - 1;
  if (x * x + y * y < 1) {
    output[idx] = 1;
  } else {
    output[idx] = 0;
  }
}

__global__ void SparkGPUPi_reduce(int *input, int *output, long size, int stage, int totalStages) {
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

