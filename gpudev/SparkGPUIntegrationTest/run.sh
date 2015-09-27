export SPARK_HOME="`pwd`/../.."

$SPARK_HOME/bin/spark-submit --class SparkGPUIntegrationTest ./target/SparkGPUIntegrationTest-1.0.jar
