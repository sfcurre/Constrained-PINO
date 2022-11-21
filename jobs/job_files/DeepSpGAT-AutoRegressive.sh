#!/bin/bash
#SBATCH --job-name=PIGNN
#SBATCH --output="out/PIGNN_DeepSpGAT-AutoRegressive-%j.out"
#SBATCH --time=23:59:59
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
python main.py --epochs 20 --patience 100 --learning-rate 0.0001 --weight-decay 1e-08\
        --hidden 32 --layers 6 --nb-heads 4 \
    --dropout 0.0 --alpha 0.2 --timesteps 5 \
    --periodic-bc \
    --model-type AutoRegressive --name DeepSpGAT-AutoRegressive

