#!/bin/bash
TIMESTAMP=$(date +%Y%m%d-%H:%M:%S)

MODELDIR=../../model/$TIMESTAMP
mkdir -p "$MODELDIR"
cp ./config/Configuration.ini $MODELDIR/Configuration.ini

../../bin/greedychunk -d "$1" -m "$MODELDIR" 1> /dev/null 2> ../../results/greedychunk-output.$TIMESTAMP
