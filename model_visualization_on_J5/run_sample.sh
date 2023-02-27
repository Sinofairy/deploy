#!/bin/bash

chmod +x start_nginx.sh
sh start_nginx.sh

export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

chmod +x ./J5_Sample/J5_Sample

log_level=$1
if [ -z "$log_level" ];then
log_level="w" #(d/i/w/e)
fi
echo "log_level: $log_level"
./J5_Sample/J5_Sample -${log_level}
