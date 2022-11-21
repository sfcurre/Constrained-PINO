# no packages

def write_bash(fp, call, name):
    with open(f'jobs/job_files/{name}.sh', 'w') as fp2:
        fp2.write(call)
    fp.write(f'sbatch {name}.sh\n')

def make_call(name, time='23:59:59', grd=None, sol=None, pdebench=None, e=100, lr=1e-4, wd=1e-8, bs=None, hdim=32, layers=6, heads=4,
              dropout=0.0, alpha=0.2, patience=100, timesteps=5, periodic=True, modeltype='PushForward', video=False):
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
    call = f'python main.py --epochs {e} --patience {patience} --learning-rate {lr} --weight-decay {wd} \\\n'
    if bs:
        call += f'    --batch-size {bs} \\\n'

    call += f'    --hidden {hdim} --layers {layers} --nb-heads {heads} \\\n'
    call += f'    --dropout {dropout} --alpha {alpha} --timesteps {timesteps} \\\n'

    if periodic:
        call += f'    --periodic-bc \\\n'

    call += f'    --model-type {modeltype} --name {name}\n\n'

    return HEADER + call, name

def main():
    with open('jobs/batch_jobs.sh', 'w') as fp:
        fp.write('cd ~/Physics-Informed-GNNs/jobs/job_files\n\n')
        write_bash(fp, *make_call('DeepSpGAT-OneStep', grd='TGV/ijk_snapshots/ijkmvmv_grd.npy', sol='TGV/ijk_snapshots/ijkmvmv_sol.npy',
                                  e=20, timesteps=1, modeltype='OneStep'))
        write_bash(fp, *make_call('DeepSpGAT-PushForward', grd='TGV/ijk_snapshots/ijkmvmv_grd.npy', sol='TGV/ijk_snapshots/ijkmvmv_sol.npy',
                                  e=20, timesteps=2, modeltype='PushForward'))
        write_bash(fp, *make_call('DeepSpGAT-AutoRegressive', grd='TGV/ijk_snapshots/ijkmvmv_grd.npy', sol='TGV/ijk_snapshots/ijkmvmv_sol.npy',
                                  e=20, timesteps=5, modeltype='AutoRegressive'))

if __name__ == '__main__':
    main()