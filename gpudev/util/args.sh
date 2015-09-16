DIR=../..
# might need to use -Dos.arch=ppc64le on OpenJDK in MAVEN_OPTS and JAVA_OPTS because of a bug
export MAVEN_OPTS="$MAVEN_OPTS -Xmx32G -XX:MaxPermSize=8G -XX:ReservedCodeCacheSize=2G"
MVN_ARGS="-Pyarn -Phadoop-2.4 -Dhadoop.version=2.4.0 -Pkinesis-asl -Phive-thriftserver -Phive"
export JAVA_OPTS="-Xmx32G -XX:MaxPermSize=8G -XX:ReservedCodeCacheSize=2G"

#export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-ppc64el"
export JAVA_HOME="/opt/ibm/ibm-java-ppc64le-80SR1FP10"
export PATH="$JAVA_HOME/bin:$PATH"

EXCL_TAGS=\
org.apache.spark.SlowTest,\
org.apache.spark.PPCIBMJDKFailingTest

MVN_COMPILE_PARALLEL_THREADS=20

MVN_CMD="./build/mvn --force"

DBG_PORT=5004

export ZINC_PORT=3032

function killZinc() {
    build/zinc-*/bin/zinc -shutdown || echo DID NOT SHUTDOWN ZINC
    sleep 3
    if ps aux | grep zinc | grep java | grep $ZINC_PORT; then
	echo ZINC IS STILL RUNNING
	return 1
    fi
}
