#!/bin/bash
source args.sh
cd $DIR

jdb -attach $DBG_PORT
