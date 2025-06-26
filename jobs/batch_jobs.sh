cd ~/Physics-Informed-GNNs/jobs/job_files

sbatch PDEBench_2D_Rand-FNO-AutoRegressive-k1-pde0.0-scaled_mass.sh
exit 1
sbatch PDEBench_2D_Rand-FNO-AutoRegressive-k1-pde0.0-scaled_momentum.sh
sbatch PDEBench_2D_Rand-FNO-AutoRegressive-k1-pde0.0-unscaled.sh
sbatch PDEBench_2D_Rand-FNO-Static-k1-pde0.0-scaled_mass.sh
sbatch PDEBench_2D_Rand-FNO-Static-k1-pde0.0-scaled_momentum.sh
sbatch PDEBench_2D_Rand-FNO-Static-k1-pde0.0-unscaled.sh
sbatch PINO_2D_Vort-FNO-AutoRegressive-k1-pde0.0-scaled_momentum.sh
sbatch PINO_2D_Vort-FNO-AutoRegressive-k1-pde0.0-unscaled.sh
sbatch PINO_2D_Vort-FNO-Static-k1-pde0.0-scaled_momentum.sh
sbatch PINO_2D_Vort-FNO-Static-k1-pde0.0-unscaled.sh
