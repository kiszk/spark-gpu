CUDA_PATH ?= /usr/local/cuda
CXX ?= g++
NVCC ?= $(CUDA_PATH)/bin/nvcc
COMPUTE_CAPABILITY ?= 35
CXXFLAGS ?= -m64 -O3 -Xcompiler -Wall --std=c++11
NVCCFLAGS ?= -arch=sm_$(COMPUTE_CAPABILITY) -Xptxas="-v"


all : cpuLR gpuLR testKernel.ptx
	

cpuLR : lrcpu.c
	cc -o cpuLR lrcpu.c -lm

gpuLR : lrgpu.cu lrgpu.h
	nvcc lrgpu.cu -o gpuLR -arch=sm_35

testKernel.ptx: testKernel.cu
	$(NVCC) -ccbin $(CXX) $(CXXFLAGS) $(NVCCFLAGS) -ptx -o $@ -c $^
	$(NVCC) testKernel.cu -o testKernel -arch=sm_35
	cp testKernel.ptx /home/kmadhu/trl-spark-gpu/core/src/test/resources

clean:
	rm -f cpuLR gpuLR lrKernel.ptx testKernel.ptx
