#!/bin/bash
source args.sh
cd $DIR

if [ 'uname -s' == "Darwin" ]; then
  OS=apple
  ARCH=x86_64
  EXT=dylib
else
  OS=linux
  ARCH=`arch`
  EXT=so
fi
DISTASM=dist/lib/spark-assembly-1.6.0-GPU-hadoop2.4.0.jar
if [ ! -f $DISTASM ]; then
  echo "file $DISTASM does not exist. Please check file name."
  exit 1
fi


function addlib2jar {
  libdir=$2/target
  lib=lib/$4-${OS}-${ARCH}.${EXT}
  if [ $1 == "DIST" ]; then
    jar=${DISTASM}
  else
    jar=$1/target/lib/$3.jar
  fi
  if [ -e $libdir/$lib ]; then
    if [ -e $jar ]; then
      jar uf $jar -C $libdir $lib
    fi 
  fi 
}

for dir in *; do
  if [ -d $dir ]; then
     echo $dir is being processed
     addlib2jar $dir $dir jcuda libJCudaDriver
     addlib2jar $dir $dir jcuda libJCudaRuntime
     for lib in ublas ufft urand usolver usparse; do
       addlib2jar $dir jc${lib} libJC${lib} 
     done
  fi
done
echo $DISTASM is being processed
addlib2jar DIST core jcuda libJCudaDriver
addlib2jar DIST core jcuda libJCudaRuntime
for lib in ublas ufft urand usolver usparse; do
  addlib2jar DIST core jc${lib} libJC${lib} 
done
