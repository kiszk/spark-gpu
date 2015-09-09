#!/bin/bash
# should run a test, but doesn't work
source args.sh
cd $DIR

SCALA_TESTS=\
org.apache.spark.ColumnPartitionSchema,\
org.apache.spark.storage.BlockManagerSuite,\
org.apache.spark.CacheManagerSuite

JAVA_TESTS=none

build/sbt -Pyarn -Phadoop-2.4 -Dhadoop.version=2.4.0 -Pkinesis-asl -Phive-thriftserver -Phive "test-only org.apache.spark.BlockManager"
