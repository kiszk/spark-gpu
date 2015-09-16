#!/bin/bash
source args.sh
cd $DIR

$MVN_CMD $MVN_ARGS -DtagsToExclude=$EXCL_TAGS --fail-never test $@ 2>&1 | tee ~/testlog-all.txt
killZinc
