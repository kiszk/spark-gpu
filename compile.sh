#!/bin/bash
# example usages: # ./compile.sh - package Spark without assembly modules (much faster) # ./compile.sh clean install - package install of Spark without assembly modules # ./compile.sh full - package everything ("normal" compilation) # ./compile.sh clean full install - package and install everything ("normal" clean install) 
ABSOLUTE_DIR=`realpath .`

# might need to use -Dos.arch=ppc64le on OpenJDK in MAVEN_OPTS and JAVA_OPTS because of a bug
export MAVEN_OPTS="$MAVEN_OPTS -Xmx32G -XX:MaxPermSize=8G -XX:ReservedCodeCacheSize=2G"
#MVN_ARGS="-Pyarn -Phadoop-2.4 -Dhadoop.version=2.4.0 -Dscala-2.11 -Pkinesis-asl -Phive-thriftserver -Phive"
MVN_ARGS="-Pyarn -Phadoop-2.4 -Dhadoop.version=2.4.0 -Dmaven.javadoc.skip=true "
export JAVA_OPTS="-Xmx32G -XX:MaxPermSize=8G -XX:ReservedCodeCacheSize=2G"

#export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-ppc64el"
export JAVA_HOME="/opt/ibm/ibm-java-ppc64le-80SR1FP10"
#export JAVA_HOME="/usr/lib/jvm/java-7-openjdk-amd64/"
export PATH="$JAVA_HOME/bin:$PATH"

MVN_CMD="./build/mvn --force"

DBG_PORT=5004

UNIQ_USER_VAL=1234
export ZINC_PORT=$((3031+UNIQ_USER_VAL))

export LD_LIBRARY_PATH="$ABSOLUTE_DIR/core/target/lib:$LD_LIBRARY_PATH"

function killZinc() {
    kill `ps aux | grep zinc | grep java | grep $ZINC_PORT | awk '{print $2}'` &&
	echo KILLED ZINC ||
	echo ZINC WAS NOT RUNNING
}

if [[ "$1" == "clean" ]]; then
    CLEAN_ARGS=clean
    shift 1
fi

if [[ !("$@" =~ "-pl") ]]; then
    if [[ "$1" == "full" ]]; then
	# used to compile everything
	SKIP_MODULES=''
	shift 1
    elif [[ "$1" == "main" ]]; then
	# used to compile main modules including assembly, which is used to launch spark externally
	SKIP_MODULES='-pl !examples,!extras/kinesis-asl-assembly,!external/flume-assembly,!external/kafka-assembly'
	shift 1
    else
	# used to compile minimum set to be able to run all tests
	# this is because assemblies and examples take a lot of time to compile
	SKIP_MODULES='-pl !examples,!assembly,!extras/kinesis-asl-assembly,!external/flume-assembly,!external/kafka-assembly'
    fi
fi
echo `pwd`
time $MVN_CMD -T 40 $MVN_ARGS $SKIP_MODULES -DskipTests $CLEAN_ARGS install $@ 2>&1 | tee ~/compile.txt
killZinc
