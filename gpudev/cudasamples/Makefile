all : cpuLR gpuLR
	

cpuLR : lrcpu.c
	cc -o cpuLR lrcpu.c -lm

gpuLR : lrgpu.cu lrgpu.h
	nvcc lrgpu.cu -o gpuLR -arch=sm_35
