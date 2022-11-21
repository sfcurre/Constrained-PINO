#!/bin/bash
#SBATCH --job-name=MeshRefinement
#SBATCH --time=3:59:59
#SBATCH --output="MeshRefinement_simple-%j.out"
#SBATCH --account=PAS2170
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL

#Code to run main.py

set -e

python main.py -X ../data/ML/601x162x241/LOW/X.npy -Y ../data/ML/601x162x241/LOW/X.npy \
    -G simple_linear -Ds ds_2x64_2x128\
    --dim 2 --tile-size 40 --upres 1 --density-minimum 0.0 --train-length 800 \
    --batch-size 128 --epochs 100 --learning-rate 0.0005 \
    --model-path ./models/saved_models/ --yx 160 576 --name simple_linear_w \
    -w -gp 1.0