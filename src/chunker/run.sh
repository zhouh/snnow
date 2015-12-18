#!/bin/bash
DATE=$(date +%Y%m%d-%H:%M:%S)
../../bin/chunk 1> /dev/null 2> output.$DATE
