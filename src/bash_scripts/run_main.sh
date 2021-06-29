#!/bin/bash

# path to anaconda environment
export PATH=/netscratch/bhatt/Environment/miniconda3/bin/:$PATH

#working dir
cd /netscratch/bhatt/Repositories/time-series-attribution/src/scripts/

python main.py  --gridEvalParams '{"percs":[99,98,96,92], "rand_layers":[0,-1,-2,-3]'} \
                --load True \
                --n_samples 1000
