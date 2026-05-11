#!/bin/sh

HOSTNAME="abgerard@manneback"
PERF_DIR="/home/ucl/inma/abgerard/LINMA2710/project/performance"

if [ $1 ]; then
    scp "$HOSTNAME:$PERF_DIR/$1.csv" performance/
    exit 0
else
    echo "Import everything from cluster? (y/n)"
    read -r answer
    if [ "$answer" != "y" ]; then
        echo "Aborting import."
        exit 0
    else
        scp "$HOSTNAME:$PERF_DIR/*.csv" performance/
    fi 
fi
