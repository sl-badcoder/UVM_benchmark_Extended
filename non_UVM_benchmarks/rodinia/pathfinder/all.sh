#!/usr/bin/env bash
exec > output_explicit_kmeans.txt 2>&1
echo "-----2-------"
./path 65536 8192  8
echo "-----4-------"
./path 65536 16384  8
echo "-----6-------"
./path 65536 24576  8
echo "-----8-------"
./path 65536 32768  8
echo "-----10-------"
./path 65536 40960  8
echo "-----12-------"
./path 65536 49152  8
echo "-----14-------"
./path 65536 57344  8