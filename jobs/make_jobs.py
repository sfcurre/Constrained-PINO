# no packages

def write_bash(fp, pred_fp, call, pred_call, name):
    with open(f'jobs/job_files/{name}.sh', 'w') as fp2:
        fp2.write(call)

    with open(f'jobs/job_files/{name}_pred.sh', 'w') as fp2:
        fp2.write(pred_call)
    
    fp.write(f'sbatch {name}.sh\n')
    pred_fp.write(f'sbatch {name}_pred.sh\n')

def make_call(name, time='23:59:59', grd=None, sol=None, pdebench=None,
              e=100, lr=1e-4, wd=1e-8, bs=None, hdim=32, layers=6, heads=4,
              dropout=0.0, alpha=0.2, pde_weight=1, patience=100, timesteps=5,
              periodic=True, use_grid=False, minibatch=0, modeltype='PushForward',
              nn_arch='PIGNN', history=0, scale_totals=True, scale_ke=False):
    HEADER = f'''#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --time={time}
#SBATCH --output="out/{name}-%j.out"
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
'''
    PRED_HEADER = f'''#!/bin/bash
#SBATCH --job-name=Pred-{name}
#SBATCH --time=0:29:59
#SBATCH --output="out/Pred-{name}-%j.out"
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
'''
    
    call = f'python main.py --epochs {e} --patience {patience} --lr {lr} --weight-decay {wd} \\\n'
    call += f'    --hidden {hdim} --layers {layers} --nb-heads {heads} --dropout {dropout} \\\n'
    call += f'    --alpha {alpha} --pde-weight {pde_weight} --timesteps {timesteps} --history {history} \\\n'
    call += f'    --pdebench {pdebench} \\\n'

    if scale_totals:
        call += f'    --scale-totals \\\n'

    if scale_ke:
        call += f'    --scale-ke \\\n'

    if bs:
        call += f'    --batch-size {bs} \\\n'

    call += f'    --model-type {modeltype} --nn-arch {nn_arch} --name {name}'

    pred_call = call + ' \\\n'
    pred_call += f'    --no-train --load-pkl out/saved_models/{name}.pkl\n\n'
    pred_call += f'python visuals_pdebench.py --name {name}\\\n'
    pred_call += f'    --run-true {pdebench}\n\n'

    call += '\n\n'

    return HEADER + call, PRED_HEADER + pred_call, name

def main():
    pdebench_1d=(("/fs/scratch/PAS2170/pdebench/data/1D/CFD/Train/"
                  "1d_cfd_rand_eta1.e-8_zeta1.e-8_periodic_train.hdf5"),
                  "1D_Rand")
    
    pdebench_2d=(("/fs/scratch/PAS2170/pdebench/data/2D/CFD/2D_Train_Rand/"
                  "2d_cfd_rand_m0.1_eta1e-08_zeta1e-08_periodic_512_train.hdf5"),
                  "2D_Rand")
    pdebench_2d_turb=(("/fs/scratch/PAS2170/pdebench/data/2D/CFD/2D_Train_Turb/"
                       "2d_cfd_turb_m0.1_eta1e-08_zeta1e-08_periodic_512_train.hdf5"),
                       "2D_Turb")

    pdebench_3d=(("/fs/scratch/PAS2170/pdebench/data/3D/Train/"
                  "3d_cfd_rand_m1.0_eta1e-08_zeta1e-08_periodic_train.hdf5"),
                  "3D_Rand")
    pdebench_3d_turb=(("/fs/scratch/PAS2170/pdebench/data/3D/Train/"
                       "3d_cfd_turb_m1.0_eta1e-08_zeta1e-08_periodic_train.hdf5"),
                       "3D_Turb")

    grd='TGV/ijk_snapshots/ijkmvmv_grd.npy'
    sol='TGV/ijk_snapshots/ijkmvmv_sol.npy'
    
    with open('jobs/batch_jobs.sh', 'w') as fp:
        with open('jobs/batch_preds.sh', 'w') as pred_fp:
            fp.write('cd ~/Physics-Informed-GNNs/jobs/job_files\n\n')
            pred_fp.write('cd ~/Physics-Informed-GNNs/jobs/job_files\n\n')
            
            for pde, name in [pdebench_2d]:#, pdebench_1d, pdebench_3d]:
                for lr in [1e-5]:
                    for pde_weight in [0.01, 0.0]: #, 0.1]:
                        for modeltype in ['OneStep', 'AutoRegressive']:
                            for scale_ke in [True, False]:  
                                write_bash(fp, pred_fp, 
                                  *make_call(f'PDEBench_{name}-{modeltype}-lr_{lr}-pde_{pde_weight}-scaled' + ('_ke' if scale_ke else ''),
                                              pdebench=pde,
                                              e=1000,
                                              lr=lr,
                                              layers=4,
                                              hdim=64,
                                              timesteps=5,
                                              pde_weight=pde_weight,
                                              modeltype=modeltype,
                                              nn_arch=f'FNO',
                                              history=10,
                                              scale_totals=True,
                                              scale_ke=scale_ke))
                            
                            write_bash(fp, pred_fp, 
                            *make_call(f'PDEBench_{name}-{modeltype}-lr_{lr}-pde_{pde_weight}-unscaled',
                                        pdebench=pde,
                                        e=1000,
                                        lr=lr,
                                        layers=4,
                                        hdim=64,
                                        timesteps=5,
                                        pde_weight=pde_weight,
                                        modeltype=modeltype,
                                        nn_arch=f'FNO',
                                        history=10,
                                        scale_totals=False))

if __name__ == '__main__':
    main()
