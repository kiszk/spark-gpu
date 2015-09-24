CUDA_PATH ?= /usr/local/cuda
CXX ?= g++
NVCC ?= $(CUDA_PATH)/bin/nvcc
COMPUTE_CAPABILITY ?= 20
CXXFLAGS ?= -O3 -Xcompiler -Wall --std=c++11
NVCCFLAGS ?= -arch=sm_$(COMPUTE_CAPABILITY) -Xptxas="-v"

all: identity.ptx

%.ptx: %.cu
	$(NVCC) -ccbin $(CXX) $(CXXFLAGS) $(NVCCFLAGS) -ptx -o $@ -c $^

clean:
	rm -f identity.ptx

.PHONY: all clean
