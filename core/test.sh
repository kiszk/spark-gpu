#!/bin/bash
mvn test -DwildcardSuites=org.apache.spark.ColumnPartitionSchema,org.apache.spark.storage.BlockManagerSuite,org.apache.spark.CacheManagerSuite -Dtest=none
