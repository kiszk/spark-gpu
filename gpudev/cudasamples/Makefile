all : cpuLR gpuLR
	

cpuLR : lr1.c
	cc -o cpuLR lr1.c -lm

gpuLR : lr1.cu
	nvcc lr1.cu -o gpuLR
