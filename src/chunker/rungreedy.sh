#!/bin/bash
DATE=$(date +%Y%m%d-%H:%M:%S)
../../bin/greedychunk "$1" 1> /dev/null 2> ../../results/greedychunk-output.$DATE
