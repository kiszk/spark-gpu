#!/bin/bash
source args.sh
cd $DIR/core
while true; do
    ../$MVN_CMD $MVN_ARGS scala:console
done
