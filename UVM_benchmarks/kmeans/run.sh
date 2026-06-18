#!/usr/bin/env bash
exec > output_our_kmeans_2.txt 2>&1

./std_kmeans --random-gib 2  32 2 2 
./std_kmeans --random-gib 4  32 2 4 
./std_kmeans --random-gib 6  32 2 6
./std_kmeans --random-gib 8  32 2 8 
./std_kmeans --random-gib 10 32 2 10 
./std_kmeans --random-gib 12 32 2 12 
./std_kmeans --random-gib 14 32 2 14 
./std_kmeans --random-gib 16 32 2 16 
./std_kmeans --random-gib 18 32 2 18 
./std_kmeans --random-gib 20 32 2 20 
./std_kmeans --random-gib 22 32 2 22 
./std_kmeans --random-gib 24 32 2 24