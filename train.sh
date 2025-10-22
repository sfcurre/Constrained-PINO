#!/bin/bash

#SBATCH -N 1                    # use 1 node
#SBATCH -n 1                    # 1 task
#SBATCH --cpus-per-task 2       # cpu cores to use
#SBATCH -t 1-00:00:00           # 1 days, 0 hours, 0 minutes, 0 seconds
#SBATCH -p gpu_batch            # use the gpu partition
#SBATCH -J CPINO                # Job name
#SBATCH --mem=32000             # 32000 MB memory (RAM)
#SBATCH --gres=gpu:1            # request 1 GPU
#SBATCH -o reports/CPINO-%A.txt
#SBATCH -e reports/CPINO-Err-%A.txt

#Code to run main.py

set -x
set -e

# module load cuda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cpino

cd ~/Constrained-PINO

# python main.py --config configs/pdebench_2d_ar.yaml
# python main.py --config configs/pdebench_2d_ar_mass.yaml
# python main.py --config configs/pdebench_2d_ar_momentum.yaml

# python main.py --config configs/pdebench_2d_ar_turb.yaml
# python main.py --config configs/pdebench_2d_ar_mass_turb.yaml
# python main.py --config configs/pdebench_2d_ar_momentum_turb.yaml

# python main.py --config configs/pdebench_2d_dilated.yaml
# python main.py --config configs/pdebench_2d_dilated_momentum.yaml

# python main.py --config configs/pdebench_2d_ar_nograd.yaml
python main.py --config configs/pdebench_2d_ar_turb_nograd.yaml