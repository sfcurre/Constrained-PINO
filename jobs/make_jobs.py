# no packages

def write_bash(fp, call, name):
    with open(f'jobs/job_files/{name}.sh', 'w') as fp2:
        fp2.write(call)
    fp.write(f'sbatch {name}.sh\n')

def make_call(name, time='23:59:59', grd=None, sol=None, pdebench=None,
              e=100, lr=1e-4, wd=1e-8, bs=None, hdim=32, layers=6, heads=4,
              dropout=0.0, alpha=0.2, pde_weight=1, patience=100, timesteps=5,
              periodic=True, use_grid=False, minibatch=0, modeltype='PushForward',
              nn_arch='PIGNN', video=False):
    HEADER = f'''#!/bin/bash
#SBATCH --job-name=PIGNN-{name}
#SBATCH --time={time}
#SBATCH --output="out/PIGNN-{name}-%j.out"
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
    call += f'    --alpha {alpha} --pde-weight {pde_weight} --timesteps {timesteps} \\\n'
    if grd and sol:
        call += f'    --grd {grd} --sol {sol} \\\n'
    elif pdebench:
        call += f'    --pdebench {pdebench} \\\n'
    else:
        raise ValueError(f"Must define data files (grd and sol or pdebench) for {name}")

    if periodic:
        call += f'    --periodic-bc \\\n'
        if minibatch:
            call += f'    --mini-batch-width {minibatch} \\\n'

    if use_grid:
        call += f'    --use-grid \\\n'

    if bs:
        call += f'    --batch-size {bs} \\\n'

    call += f'    --model-type {modeltype} --nn-arch{nn_arch} --name {name}\n\n'

    return HEADER + call, name

def main():
    pdebench=("/fs/scratch/PAS2170/pdebench/data/2D/CFD/2D_Train_Rand/"
              "2d_cfd_rand_m0.1_eta1e-08_zeta1e-08_periodic_512_train.hdf5")

    grd='TGV/ijk_snapshots/ijkmvmv_grd.npy'
    sol='TGV/ijk_snapshots/ijkmvmv_sol.npy'
    
    with open('jobs/batch_jobs.sh', 'w') as fp:
        fp.write('cd ~/Physics-Informed-GNNs/jobs/job_files\n\n')
        
        for lr in [1e-4, 1e-5]:
            for pde_weight in [0.0]:
                for modeltype in ['OneStep', 'PushForward', 'AutoRegressive']:

                    write_bash(fp, *make_call(f'PDEBench2D-{modeltype}-lr_{lr}-pde_{pde_weight}',
                                            pdebench=pdebench,
                                            e=100,
                                            lr=lr,
                                            layers=4,
                                            hdim=64,
                                            timesteps=5,
                                            pde_weight=pde_weight,
                                            modeltype=modeltype,
                                            nn_arch='FNO2d',
                                            use_grid=True))

if __name__ == '__main__':
    main()
