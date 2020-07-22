#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $(readlink -f $0 || echo $0));pwd -P)
source $SCRIPT_DIR/args.sh
cd $DIR

jdb -attach $DBG_PORT
