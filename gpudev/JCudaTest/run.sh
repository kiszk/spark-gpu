#!/bin/bash
#LD_LIBRARY_PATH=/usr/local/cuda-7.0/targets/ppc64le-linux/lib
# can also run with mvn exec:exec

LD_LIBRARY_PATH=target/lib scala -classpath "/home/janw/.m2/repository/org/mystic/mavenized-jcuda/0.1.2/mavenized-jcuda-0.1.2.jar:/home/janw/.m2/repository/jcuda/jcuda/0.7.0a/jcuda-0.7.0a.jar:target/JCudaTest-1.0.jar" JCudaTest
