#!/bin/bash
LD_LIBRARY_PATH=target/lib scala -classpath target/lib/jcuda.jar:target/StreamBenchmark-1.0.jar StreamBenchmark $@
