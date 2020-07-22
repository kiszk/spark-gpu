CUDA_PATH ?= /usr/local/cuda
CXX ?= g++
NVCC ?= $(CUDA_PATH)/bin/nvcc
COMPUTE_CAPABILITY ?= 35
CXXFLAGS ?= -m64 -O3 -Xcompiler -Wall --std=c++11
NVCCFLAGS ?= -arch=sm_$(COMPUTE_CAPABILITY) -Xptxas="-v"

all: target/JCudaTest-1.0.jar

target/JCudaTest-1.0.jar: src/main/scala/JCudaTest.scala src/main/resources/sum.cubin
	mvn package

src/main/resources/sum.cubin: src/cuda/sum.cu
	mkdir -p src/main/resources
	$(NVCC) -ccbin $(CXX) $(CXXFLAGS) $(NVCCFLAGS) -cubin -o $@ -c $^

run: run.sh target/JCudaTest-1.0.jar
	bash run.sh

clean:
	mvn clean
	rm -f src/main/resources/sum.cubin
