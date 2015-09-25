#!/bin/bash
source args.sh
cd $DIR/core
trap '' 2
export LD_LIBRARY_PATH="target/lib:$LD_LIBRARY_PATH"
../$MVN_CMD $MVN_ARGS scala:console
