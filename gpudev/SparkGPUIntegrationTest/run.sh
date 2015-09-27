export SPARK_HOME="`pwd`/../.."

export LD_LIBRARY_PATH="`pwd`/target/lib:$LD_LIBRARY_PATH"
$SPARK_HOME/bin/spark-submit --class SparkGPUIntegrationTest ./target/SparkGPUIntegrationTest-1.0.jar
