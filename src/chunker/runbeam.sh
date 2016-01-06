#!/bin/bash
DATE=$(date +%Y%m%d-%H:%M:%S)
../../bin/beamchunk   "$1" 1> /dev/null 2> ../../results/beamchunk-output.$DATE
