#!/bin/bash
# run only the tests being worked on
# ./testdev.sh compile also compiles the stuff before
# ./testdev.sh debug waits for attachment of a scala debugger
source args.sh
cd $DIR

SCALA_TESTS=\
org.apache.spark.IteratedPartitionDataSuite,\
org.apache.spark.ColumnPartitionSchemaSuite,\
org.apache.spark.ColumnPartitionDataSuite,\
org.apache.spark.ColumnPartitionDataBuilderSuite,\
org.apache.spark.storage.BlockManagerSuite,\
org.apache.spark.rdd.RDDSuite,\
org.apache.spark.CacheManagerSuite

#org.apache.spark.DistributedSuite,\
#org.apache.spark.rdd.AsyncRDDActionsSuite,\
#org.apache.spark.rdd.PipedRDDSuite,\
#org.apache.spark.rdd.DoubleRDDSuite,\
#org.apache.spark.rdd.JdbcRDDSuite,\
#org.apache.spark.rdd.RDDOperationScopeSuite,\
#org.apache.spark.rdd.RDDSuiteUtils,\
#org.apache.spark.rdd.PartitionwiseSampledRDDSuite,\
#org.apache.spark.rdd.SortingSuite,\
#org.apache.spark.rdd.ParallelCollectionSplitSuite,\
#org.apache.spark.rdd.RDDSuite,\
#org.apache.spark.rdd.PairRDDFunctionsSuite,\
#org.apache.spark.rdd.LocalCheckpointSuite,\
#org.apache.spark.rdd.PartitionPruningRDDSuite,\
#org.apache.spark.rdd.ZippedPartitionsSuite,\
#org.apache.spark.rdd.PartitionPruningRDDSuite,\
#org.apache.spark.rdd.PipedRDDSuite,\
#org.apache.spark.rdd.RDDOperationScopeSuite,\
#org.apache.spark.rdd.RDDSuite,\
#org.apache.spark.scheduler.DAGSchedulerSuite,\
#org.apache.spark.CheckpointSuite,\

JAVA_TESTS=\
org.apache.spark.unsafe.memory.ExecutorMemoryManagerSuite

MODULES=unsafe,core

if [[ "$1" == "compile" ]]; then
    COMPILE=true
    shift 1
fi

if [[ "$1" == "debug" ]]; then
    DBG_ARGS="-DforkMode=once -DdebugForkedProcess=true -DdebuggerPort=$DBG_PORT"
    shift 1
fi

if [ $COMPILE ]; then
    ./compile.sh -pl $MODULES $@
fi

$MVN_CMD $MVN_ARGS -DtagsToExclude=$EXCL_TAGS --fail-at-end test -DwildcardSuites=$SCALA_TESTS -Dtest=$JAVA_TESTS -pl $MODULES $DBG_ARGS $@ 2>&1 | tee ~/testlog-dev.txt
killZinc
