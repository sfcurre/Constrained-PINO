#!/bin/bash
#SBATCH --job-name=MeshRefinement
#SBATCH --time=23:59:59
#SBATCH --output="MeshRefinement_resnet_2x64_2x128-%j.out"
#SBATCH --account=PAS2170
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL

#Code to run main.py

set -x
set -e

module load cuda
source /usr/local/python/3.6-conda5.2/etc/profile.d/conda.sh
conda activate deepml4

cd ~/ML-in-FluidMechanics/mesh_refinement

python main.py -X ../data/ML/601x162x241/LOW/ -Y ../data/ML/1201x324x481/HIGH/ \
    -G resnet_2x64_2x128 \
    --dim 2 --tile-size 32 --upres 1 --density-minimum 0.02 --train-length 800 \
    --coarsen --data-augmentation --batch-size 1 --epochs 10 --learning-rate 0.0002 \
    --output ./output_t1.npy --model-path ./models/saved_models/ --yx 160 600
    
python main.py -X ../data/ML/601x162x241/LOW/ -Y ../data/ML/1201x324x481/HIGH/ \
    -G resnet_2x64_2x128 -Ds ds_2x64_2x128 \
    --dim 2 --tile-size 32 --upres 1 --density-minimum 0.02 --train-length 800 \
    --coarsen --data-augmentation --batch-size 1 --epochs 10 --learning-rate 0.0002 \
    --output ./output_t2.npy --model-path ./models/saved_models/ --yx 160 600
    
python main.py -X ../data/ML/601x162x241/LOW/ -Y ../data/ML/1201x324x481/HIGH/ \
    -G resnet_2x64_2x128 -Ds ds_2x64_2x128 -Dt dt_2x64_2x128 \
    --dim 2 --tile-size 32 --upres 1 --density-minimum 0.02 --train-length 800 \
    --coarsen --data-augmentation --batch-size 1 --epochs 10 --learning-rate 0.0002 \
    --output ./output_t3.npy --model-path ./models/saved_models/ --yx 160 600
    
python main.py -X ../data/ML/601x162x241/LOW/ -Y ../data/ML/1201x324x481/HIGH/ \
    -G resnet_2x64_2x128 -Dt dt_2x64_2x128 \
    --dim 2 --tile-size 32 --upres 1 --density-minimum 0.02 --train-length 800 \
    --coarsen --data-augmentation --batch-size 1 --epochs 10 --learning-rate 0.0002 \
    --output ./output_t4.npy --model-path ./models/saved_models/ --yx 160 600
    
python main.py -X ../data/ML/601x162x241/LOW/ -Y ../data/ML/1201x324x481/HIGH/ \
    -G resnet_2x64_2x128 -Ds ds_2x64_2x128 -Dt dt_2x64_2x128 \
    --dim 2 --tile-size 32 --upres 1 --density-minimum 0.02 --train-length 800 \
    --coarsen --batch-size 1 --epochs 10 --learning-rate 0.0002 \
    --output ./output_t5.npy --model-path ./models/saved_models/ --yx 160 600

python main.py -X ../data/ML/601x162x241/LOW/ -Y ../data/ML/1201x324x481/HIGH/ \
    -G resnet_2x64_2x128_u32 -Ds ds_2x64_2x128 -Dt dt_2x64_2x128 \
    --dim 2 --tile-size 32 --upres 2 --density-minimum 0.02 --train-length 800 \
    --data-augmentation --batch-size 1 --epochs 10 --learning-rate 0.0002 \
    --output ./output_t6.npy --model-path ./models/saved_models/ --yx 160 600

python main.py -X ../data/ML/601x162x241/LOW/ -Y ../data/ML/1201x324x481/HIGH/ \
    -G resnet_2x64_2x128 --use-gradients \
    --dim 2 --tile-size 32 --upres 1 --density-minimum 0.02 --train-length 800 \
    --coarsen --data-augmentation --batch-size 1 --epochs 10 --learning-rate 0.0002 \
    --output ./output_t7.npy --model-path ./models/saved_models/ --yx 160 600

python main.py -X ../data/ML/601x162x241/LOW/X.npy -Y ../data/ML/1201x324x481/HIGH/Y.npy \
    -G resnet_2x64_2x128 --use-prior -Ds ds_2x64_2x128 \
    --dim 2 --tile-size 64 --upres 1 --density-minimum 0.02 --train-length 800 \
    --coarsen --data-augmentation --batch-size 64 --epochs 10 --learning-rate 0.0002 \
    --output ./output_t9.npy --model-path ./models/saved_models/ --yx 160 600