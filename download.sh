#!/bin/bash

#SBATCH -N 1                    # use 1 node
#SBATCH -n 1                    # 1 task
#SBATCH --cpus-per-task 4       # cpu cores to use
#SBATCH -t 1-00:00:00           # 1 days, 0 hours, 0 minutes, 0 seconds
#SBATCH -p cpu_batch            # use the gpu partition
#SBATCH -J CPINO_Download       # Job name
#SBATCH --mem=32000             # 32000 MB memory (RAM)
#SBATCH -o reports/Download-%A.txt
#SBATCH -e reports/Download-Err-%A.txt

#Code to run main.py

set -x
set -e

# module load cuda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cpino

cd ~/Constrained-PINO

cp PDEBench/pdebench/data_download/download_metadata.csv ./

#python download_direct_custom.py --root_folder ./data/ --file_name "2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5"
#python download_direct_custom.py --root_folder /fs/scratch/PAS2170/pdebench/data/ \
#    --pde_name 1D_CFD
python download_direct_custom.py --root_folder /local/storage/current.33/cfd/ \
    --pde_name 2D_CFD
python download_direct_custom.py --root_folder /local/storage/current.33/cfd/ \
    --pde_name 3D_CFD
python download_direct_custom.py --root_folder /local/storage/current.33/cfd/ \
    --pde_name NS_Incom
    #--file_name "2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5" \
    #--file_name "2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5"