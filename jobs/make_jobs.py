# no packages

def write_bash(fp, call, name):
    with open(f'jobs/job_files/{name}.sh', 'w') as fp2:
        fp2.write(call)
    fp.write(f'sbatch {name}.sh\n')

def make_call(name, time='23:59:59', grd=None, sol=None, pdebench=None,
              e=100, lr=1e-4, wd=1e-8, bs=None, hdim=32, layers=6, heads=4,
              dropout=0.0, alpha=0.2, pde_weight=1, patience=100, timesteps=5,
              periodic=True, minibatch=0, modeltype='PushForward', video=False):
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
    if bs:
        call += f'    --batch-size {bs} \\\n'

    call += f'    --model-type {modeltype} --name {name}\n\n'

    return HEADER + call, name

def main():
    with open('jobs/batch_jobs.sh', 'w') as fp:
        fp.write('cd ~/Physics-Informed-GNNs/jobs/job_files\n\n')
        write_bash(fp, *make_call('DeepSpGAT-OneStep-0_1',
                                  grd='TGV/ijk_snapshots/ijkmvmv_grd.npy',
                                  sol='TGV/ijk_snapshots/ijkmvmv_sol.npy',
                                  e=20,
                                  lr=1e-4,
                                  layers=4,
                                  hdim=32,
                                  #heads=4,
                                  timesteps=1,
                                  minibatch=3,
                                  #bs=4,
                                  pde_weight=0.1,
                                  modeltype='OneStep'))
        write_bash(fp, *make_call('DeepSpGAT-OneStep-1_0',
                                  grd='TGV/ijk_snapshots/ijkmvmv_grd.npy',
                                  sol='TGV/ijk_snapshots/ijkmvmv_sol.npy',
                                  e=20,
                                  lr=1e-4,
                                  layers=4,
                                  hdim=32,
                                  #heads=4,
                                  timesteps=1,
                                  minibatch=3,
                                  #bs=4,
                                  pde_weight=1.0,
                                  modeltype='OneStep'))
        write_bash(fp, *make_call('DeepSpGAT-OneStep-0_01',
                                  grd='TGV/ijk_snapshots/ijkmvmv_grd.npy',
                                  sol='TGV/ijk_snapshots/ijkmvmv_sol.npy',
                                  e=20,
                                  lr=1e-4,
                                  layers=4,
                                  hdim=32,
                                  #heads=4,
                                  timesteps=1,
                                  minibatch=3,
                                  #bs=4,
                                  pde_weight=0.01,
                                  modeltype='OneStep'))
        write_bash(fp, *make_call('DeepSpGAT-PushForward',
                                  grd='TGV/ijk_snapshots/ijkmvmv_grd.npy',
                                  sol='TGV/ijk_snapshots/ijkmvmv_sol.npy',
                                  e=20,
                                  lr=1e-4,
                                  layers=4,
                                  hdim=32,
                                  #heads=4,
                                  timesteps=2,
                                  minibatch=3,
                                  #bs=4,
                                  pde_weight=0.1,
                                  modeltype='PushForward'))
        write_bash(fp, *make_call('DeepSpGAT-AutoRegressive',
                                  grd='TGV/ijk_snapshots/ijkmvmv_grd.npy',
                                  sol='TGV/ijk_snapshots/ijkmvmv_sol.npy',
                                  e=20,
                                  lr=1e-4,
                                  layers=4,
                                  hdim=32,
                                  #heads=4,
                                  timesteps=3,
                                  minibatch=3,
                                  #bs=4,
                                  pde_weight=0.1,
                                  modeltype='AutoRegressive'))

if __name__ == '__main__':
    main()