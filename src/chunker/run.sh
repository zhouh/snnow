#!/bin/bash
DATE=$(date +%Y%m%d-%H:%M:%S)
../../bin/chunk "$1" 1> /dev/null 2> ../../results/output.$DATE
