export SPARK_HOME="`pwd`/../.."

$SPARK_HOME/bin/spark-submit --class ParallelizeTest ./target/ParallelizeTest-1.0.jar
