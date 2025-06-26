import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

def plot_1d(data, name, vmin=None, vmax=None):
    fig, axes = plt.subplots(3, 1, figsize=(4, 6))
    data = np.squeeze(data)    

    if vmin is None:
        vmin = np.min(data, axis=(0, 1))
    if vmax is None:
        vmax = np.max(data, axis=(0, 1))

    for i in range(3):
        pl = axes[i].imshow(data.squeeze()[..., i], cmap='jet', origin='lower', vmin=vmin[i], vmax=vmax[i])
        fig.colorbar(pl, ax=axes[i], location='bottom')

    fig.suptitle('Fluid Channels (density, velocity, pressure)')
    plt.savefig(f'out/visuals/{name}_channels.png')
    del fig


def main():

    files = ['out/output_for_visuals/GT_testing_1D.npy',
            'out/output_for_visuals/1D/PDEBench_1D_Rand-AutoRegressive-lr_1e-05-pde_0.0-scaled_test.npy',
            'out/output_for_visuals/1D/PDEBench_1D_Rand-AutoRegressive-lr_1e-05-pde_0.0-unscaled_test.npy']

    solutions = [np.load(f).reshape(-1, 1024, 3) for f in files]
    solutions[0] = solutions[0][1:]

    errors = [solutions[1] - solutions[0], np.abs(solutions[2] - solutions[0])]

    labels = ['Ground Truth', 'Scaled', 'Unscaled']
    channels = ['Density', 'Velocity', 'Pressure']

    for i, ch in enumerate(channels):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        for j, sol in enumerate(solutions):
            if j == 0:
                vmin, vmax = sol[..., i].min(), sol[..., i].max()

            im = axes[j].imshow(sol[..., i], cmap='jet', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')        
            axes[j].set_ylabel(labels[j], fontsize=24)
            axes[j].set_xticks([])
            axes[j].set_yticks([])

        fig.colorbar(im, ax=axes.ravel().tolist())

        fig.suptitle(f'AutoRegressive Results for 1D CFD ({ch})' + ' '*12, fontsize=28)
        fig.supxlabel('Location' + ' '*12, fontsize=28)
        fig.supylabel('Time', fontsize=28)
        # plt.subplots_adjust(left=0.07)
        plt.savefig(f'visuals/1D_{ch}.png', bbox_inches='tight')

        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        vmin = min(err[..., i].min() for err in errors)
        vmax = max(err[..., i].max() for err in errors)

        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax

        for j, err in enumerate(errors):
            
            im = axes[j].imshow(err[..., i], cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')        
            axes[j].set_ylabel(labels[j+1], fontsize=24)
            axes[j].set_xticks([])
            axes[j].set_yticks([])

        fig.colorbar(im, ax=axes.ravel().tolist())
        
        fig.suptitle(f'AutoRegressive Errors for 1D CFD ({ch})' + ' '*12, fontsize=28)
        fig.supxlabel('Location' + ' '*12, fontsize=28)
        fig.supylabel('Time', fontsize=28)
        plt.savefig(f'visuals/1D_{ch}_error.png', bbox_inches='tight')



if __name__ == '__main__':
    main()