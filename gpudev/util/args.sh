DIR=../..
#ABSOLUTE_DIR=`realpath $DIR`
ABSOLUTE_DIR=`readlink -e $DIR`
# might need to use -Dos.arch=ppc64le on OpenJDK in MAVEN_OPTS and JAVA_OPTS because of a bug
export MAVEN_OPTS="$MAVEN_OPTS -Xmx32G -XX:MaxPermSize=8G -XX:ReservedCodeCacheSize=2G"
MVN_ARGS="-Pyarn -Phadoop-2.4 -Dhadoop.version=2.4.0 -Dscala-2.11 -Pkinesis-asl -Phive-thriftserver -Phive"
export JAVA_OPTS="-Xmx32G -XX:MaxPermSize=8G -XX:ReservedCodeCacheSize=2G"

#export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-ppc64el"
#export JAVA_HOME="/opt/ibm/ibm-java-ppc64le-80SR1FP10"
if [ "${JAVA_HOME}" = "" ]; then
  echo "please set an environment variable JAVA_HOME"
  exit 1
fi
export PATH="$JAVA_HOME/bin:$PATH"

EXCL_TAGS=\
org.apache.spark.SlowTest,\
org.apache.spark.PPCIBMJDKFailingTest

NCORES=`lscpu | awk ' /^Core\(s\)/ { print $4 }'`
NSOCKETS=`lscpu | awk ' /^Socket\(s\)/ { print $2 }'`
MVN_COMPILE_PARALLEL_THREADS=`expr $NCORES \* $NSOCKETS`

MVN_CMD="./build/mvn --force"

DBG_PORT=5004

export ZINC_PORT=$(python -S -c "import random; print random.randrange(3030,4030)")

export LD_LIBRARY_PATH="$ABSOLUTE_DIR/core/target/lib:$LD_LIBRARY_PATH"

function killZinc() {
    kill `ps aux | grep zinc | grep java | grep $ZINC_PORT | awk '{print $2}'` &&
	echo KILLED ZINC ||
	echo ZINC WAS NOT RUNNING
}
