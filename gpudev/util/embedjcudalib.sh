#!/bin/bash
source args.sh
cd $DIR

usage() {
  echo "Usage: $0 [-a ARCH] [-o OS] [-e EXTENSION] [-d HADOOP] [-v VERSION]" 1>&2
  exit 1
}

if [ 'uname -s' == "Darwin" ]; then
  OS=apple
  ARCH=x86_64
  EXT=dylib
else
  OS=linux
  ARCH=`arch`
  EXT=so
fi
if [ $ARCH == "ppc64le" ]; then
  ARCH="ppc_64"
fi
VERSION="1.6.0-GPU"
HADOOP="2.4.0"

while getopts a:o:e:d:v:h OPT
do
  case $OPT in
    a) ARCH=$OPTARG
       ;;
    o) OS=$OPTARG
       ;;
    e) EXT=$OPTARG
       ;;
    d) HADOOP=$OPTARG
       ;;
    v) VERSION=$OPTARG
       ;;
    h) usage
       ;;
  esac
done
shift $((OPTIND - 1))

DISTASM=dist/lib/spark-assembly-${VERSION}-hadoop${HADOOP}.jar
if [ ! -f $DISTASM ]; then
  echo "file $DISTASM does not exist. Please check file name."
  exit 1
fi


function addlib2jar {
  libdir=$2/target
  lib=lib/$4-${OS}-${ARCH}.${EXT}
  if [ ! -f $libdir/$lib ]; then
    echo "file $libdir/$lib does not exist. Please check file name."
    exit 1
  fi
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
  if [ -d $dir ] && [ -f $dir/target/lib/libJCudaDrive*.so ]; then
     echo $dir is being processed
     addlib2jar $dir $dir jcuda libJCudaDriver
     addlib2jar $dir $dir jcuda libJCudaRuntime
     for lib in ublas ufft urand usolver usparse; do
       addlib2jar $dir $dir jc${lib} libJC${lib} 
     done
  fi
done
echo $DISTASM is being processed
addlib2jar DIST core jcuda libJCudaDriver
addlib2jar DIST core jcuda libJCudaRuntime
for lib in ublas ufft urand usolver usparse; do
  addlib2jar DIST core jc${lib} libJC${lib} 
done
