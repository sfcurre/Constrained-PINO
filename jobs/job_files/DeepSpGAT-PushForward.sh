#!/bin/bash
#SBATCH --job-name=PIGNN-DeepSpGAT-PushForward
#SBATCH --time=23:59:59
#SBATCH --output="out/PIGNN-DeepSpGAT-PushForward-%j.out"
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
conda activate pignn

cd ~/Physics-Informed-GNNs
python main.py --epochs 20 --patience 100 --lr 0.0001 --weight-decay 1e-08 \
    --hidden 32 --layers 6 --nb-heads 4 \
    --dropout 0.0 --alpha 0.2 --timesteps 2 \
    --grd TGV/ijk_snapshots/ijkmvmv_grd.npy --sol TGV/ijk_snapshots/ijkmvmv_sol.npy \
    --periodic-bc \
    --model-type PushForward --name DeepSpGAT-PushForward

