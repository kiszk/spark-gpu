#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $(readlink -f $0 || echo $0));pwd -P)
source $SCRIPT_DIR/args.sh
cd $DIR/core
trap '' 2
export LD_LIBRARY_PATH="target/lib:$LD_LIBRARY_PATH"
../$MVN_CMD $MVN_ARGS scala:console
