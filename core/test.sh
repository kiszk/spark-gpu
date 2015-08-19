#!/bin/bash
mvn -Pyarn -Phadoop-2.3 -Dhadoop.version=2.3.0 -Phive -Pkinesis-asl -Phive-thriftserver --fail-at-end test -DwildcardSuites=org.apache.spark.ColumnPartitionSchema,org.apache.spark.storage.BlockManagerSuite,org.apache.spark.CacheManagerSuite -Dtest=none
