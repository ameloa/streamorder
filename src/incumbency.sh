#!/bin/bash

trap "exit" SIGINT

filename=$1
echo filename = $filename

> ../results/${filename}_c_finalFrac.txt

for threshold in {-10..10}
do
    echo threshold = $threshold
    python3 incumbency_data.py --edge_list $filename --c $threshold >> ../results/${filename}_c_finalFrac.txt &
done
