#!/usr/bin/env bash
exec > output_explicit_kmeans.txt 2>&1

./std_kmeans --random-gib 2  32 1 2 
./std_kmeans --random-gib 4  32 1 4 
./std_kmeans --random-gib 6  32 1 6
./std_kmeans --random-gib 8  32 1 8 
./std_kmeans --random-gib 10 32 1 10 
./std_kmeans --random-gib 12 32 1 12 
./std_kmeans --random-gib 14 32 1 14 