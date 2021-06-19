#!/bin/bash

# path to anaconda environment
export PATH=/netscratch/bhatt/Environment/miniconda3/bin/:$PATH

#working dir
cd /netscratch/bhatt/Repositories/time-series-attribution/src/scripts/

python main.py  --dataset "UWaveGestureLibraryAll" \
                        --load True \
                        --n_samples 1000  \
                        --gridEvalParams '{"methods" : ["Saliency","GradCAMpp","SmoothGradCAMpp"],  "approaches" : ["replaceWithMean", "replaceWithInterp"],  "percs" : [99,92,4]}'