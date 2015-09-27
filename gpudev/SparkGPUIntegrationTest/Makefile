CUDA_PATH ?= /usr/local/cuda
CXX ?= g++
NVCC ?= $(CUDA_PATH)/bin/nvcc
COMPUTE_CAPABILITY ?= 35
CXXFLAGS ?= -m64 -O3 -Xcompiler -Wall --std=c++11
NVCCFLAGS ?= -arch=sm_$(COMPUTE_CAPABILITY) -Xptxas="-v"

all: target/SparkGPUIntegrationTest-1.0.jar

target/SparkGPUIntegrationTest-1.0.jar: src/main/scala/SparkGPUIntegrationTest.scala src/main/resources/kernel.ptx
	mvn package

src/main/resources/kernel.ptx: src/cuda/kernel.cu
	mkdir -p src/main/resources
	$(NVCC) -ccbin $(CXX) $(CXXFLAGS) $(NVCCFLAGS) -ptx -o $@ -c $^

run: target/SparkGPUIntegrationTest-1.0.jar run.sh
	bash run.sh

clean:
	mvn clean
	rm -f src/main/resources/kernel.ptx
