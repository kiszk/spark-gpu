#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $(readlink -f $0 || echo $0));pwd -P)
source $SCRIPT_DIR/args.sh
cd $DIR

$MVN_CMD $MVN_ARGS -DtagsToExclude=$EXCL_TAGS --fail-never test $@ 2>&1 | tee ~/testlog-all.txt
killZinc
