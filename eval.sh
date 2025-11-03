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

python main.py --config configs/pdebench_2d_ar.yaml --load out/saved_models/pdebench_2d_ar_best.pkl --eval
python main.py --config configs/pdebench_2d_ar_mass.yaml --load out/saved_models/pdebench_2d_ar_mass_best.pkl --eval
python main.py --config configs/pdebench_2d_ar_conservative.yaml --load out/saved_models/pdebench_2d_ar_conservative_best.pkl --eval
python main.py --config configs/pdebench_2d_ar_momentum.yaml --load out/saved_models/pdebench_2d_ar_momentum_best.pkl --eval

python main.py --config configs/pdebench_2d_ar_dpot.yaml --load out/saved_models/pdebench_2d_ar_dpot_best.pkl --eval
python main.py --config configs/pdebench_2d_ar_dpot_mass.yaml --load out/saved_models/pdebench_2d_ar_dpot_mass_best.pkl --eval
python main.py --config configs/pdebench_2d_ar_dpot_conservative.yaml --load out/saved_models/pdebench_2d_ar_dpot_conservative_best.pkl --eval
python main.py --config configs/pdebench_2d_ar_dpot_momentum.yaml --load out/saved_models/pdebench_2d_ar_dpot_momentum_best.pkl --eval

python main.py --config configs/pdebench_2d_ar_dpot_Ti.yaml --load out/saved_models/pdebench_2d_ar_dpot_Ti_best.pkl --eval
python main.py --config configs/pdebench_2d_ar_dpot_Ti_mass.yaml --load out/saved_models/pdebench_2d_ar_dpot_Ti_mass_best.pkl --eval
python main.py --config configs/pdebench_2d_ar_dpot_Ti_conservative.yaml --load out/saved_models/pdebench_2d_ar_dpot_Ti_conservative_best.pkl --eval
python main.py --config configs/pdebench_2d_ar_dpot_Ti_momentum.yaml --load out/saved_models/pdebench_2d_ar_dpot_Ti_momentum_best.pkl --eval

python main.py --config configs/pdebench_2d_ar_dpot_S.yaml --load out/saved_models/pdebench_2d_ar_dpot_S_best.pkl --eval
python main.py --config configs/pdebench_2d_ar_dpot_S_mass.yaml --load out/saved_models/pdebench_2d_ar_dpot_S_mass_best.pkl --eval
python main.py --config configs/pdebench_2d_ar_dpot_S_conservative.yaml --load out/saved_models/pdebench_2d_ar_dpot_S_conservative_best.pkl --eval
python main.py --config configs/pdebench_2d_ar_dpot_S_momentum.yaml --load out/saved_models/pdebench_2d_ar_dpot_S_momentum_best.pkl --eval

# python main.py --config configs/pdebench_2d_ar_dpot_Ti_base.yaml --load out/saved_models/pdebench_2d_ar_dpot_Ti_base_best.pkl --eval
