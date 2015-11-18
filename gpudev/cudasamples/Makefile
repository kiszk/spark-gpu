CUDA_PATH ?= /usr/local/cuda
CXX ?= g++
NVCC ?= $(CUDA_PATH)/bin/nvcc
COMPUTE_CAPABILITY ?= 35
CXXFLAGS ?= -m64 -O3 -Xcompiler -Wall --std=c++11
NVCCFLAGS ?= -arch=sm_$(COMPUTE_CAPABILITY) -Xptxas="-v"


all : cpuLR gpuLR kernel.ptx
	

cpuLR : lrcpu.c
	cc -o cpuLR lrcpu.c -lm

gpuLR : lrgpu.cu lrgpu.h
	nvcc lrgpu.cu -o gpuLR -arch=sm_35

kernel.ptx: lrgpu.cu
	$(NVCC) -ccbin $(CXX) $(CXXFLAGS) $(NVCCFLAGS) -ptx -o $@ -c $^

clean:
	rm -f cpuLR gpuLR kernel.ptx
