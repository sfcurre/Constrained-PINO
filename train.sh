#!/bin/bash
#SBATCH --job-name=PINO
#SBATCH --time=23:59:59
#SBATCH --output="reports/PINO-%j.out"
#SBATCH --account=PAS2170
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL

#Code to run main.py

set -x
set -e

module load miniconda3/24.1.2-py310 cuda/12.3.0
conda activate cpino
conda list

cd ~/Physics-Informed-GNNs

python main.py --config configs/pdebench_2d.yaml

