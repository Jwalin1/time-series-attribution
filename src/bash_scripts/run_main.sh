#!/bin/bash

# path to anaconda environment
export PATH=/netscratch/bhatt/Environment/miniconda3/bin/:$PATH

#working dir
cd /netscratch/bhatt/Repositories/time-series-attribution/src/scripts/

python main.py  --dataset "SyntheticAnomaly" \
                        --load True \
                        --n_samples 1000