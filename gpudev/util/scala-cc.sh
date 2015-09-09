#!/bin/bash
source args.sh
cd $DIR

if [[ "$1" != "" ]]; then
    MODULE="$1"
    shift 1
else
    MODULE=core
fi

$MVN_CMD $MVN_ARGS scala:cc -pl $MODULE $@
