#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $(readlink -f $0 || echo $0));pwd -P)
source $SCRIPT_DIR/args.sh
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
VERSION="2.0.0-GPU"
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


function addlib2jar {
  libdir=$2/target
  lib=lib/$4-${OS}-${ARCH}.${EXT}
  if [ ! -f "$libdir/$lib" ]; then
    #echo "file $libdir/$lib does not exist. Please check file name." >&2
    return 255
  fi
  if [ "$5" == "on" ]; then
    echo "$1 is being processed" >& 2
  fi
  if [ $1 == "DIST" ]; then
    jar=${DISTASM}
  else
    jar=$1/target/lib/$3.jar
  fi
  if [ -e $libdir/$lib ]; then
    if [ -e $jar ]; then
      jar uf $jar -C $libdir $lib && echo "  added $lib" >&2
    fi 
  fi
  return 1 
}

for dir in core unsafe; do
  if [ -d $dir ] && [ $dir != "dist" ]; then
    addlib2jar $dir $dir jcuda libJCudaDriver on
    if [ $? == 1 ]; then
      addlib2jar $dir $dir jcuda libJCudaRuntime
      for lib in ublas ufft urand usolver usparse; do
        addlib2jar $dir $dir jc${lib} libJC${lib} 
      done
    fi
  fi
done

DISTASM=dist/lib/spark-assembly-${VERSION}-hadoop${HADOOP}.jar
if [ -f $DISTASM ]; then
  addlib2jar DIST core jcuda libJCudaDriver on
  if [ $? == 1 ]; then
    addlib2jar DIST core jcuda libJCudaRuntime
    for lib in ublas ufft urand usolver usparse; do
      addlib2jar DIST core jc${lib} libJC${lib} 
    done
  fi
else
  echo "file $DISTASM does not exist. Please check file name." >&2
fi

