# Makefile to generate native test code. Needs to be manually re-ran each time it changes.

CUDA_PATH ?= /usr/local/cuda
CXX ?= g++
NVCC ?= $(CUDA_PATH)/bin/nvcc
COMPUTE_CAPABILITY ?= 20
CXXFLAGS ?= -O3 -Xcompiler -Wall --std=c++11
NVCCFLAGS ?= -arch=sm_$(COMPUTE_CAPABILITY) -Xptxas="-v" --use_fast_math

TARGETS = SparkGPUExamples.ptx

all: $(TARGETS)

%.ptx: %.cu
	$(NVCC) -ccbin $(CXX) $(CXXFLAGS) $(NVCCFLAGS) -ptx -o $@ -c $^

%.cubin: %.cu
	$(NVCC) -ccbin $(CXX) $(CXXFLAGS) $(NVCCFLAGS) -cubin -o $@ -c $^

clean:
	rm -f $(TARGETS)

.PHONY: all clean
