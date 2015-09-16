#!/bin/bash
source args.sh
cd $DIR

$MVN_CMD $MVN_ARGS -DtagsToExclude=$EXCL_TAGS --fail-at-end test -pl core $@ 2>&1 | tee ~/testlog-core.txt
killZinc
