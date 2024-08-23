import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

def curl(vfield, dim=2):
    grads = np.gradient(vfield, axis=(0, 1))
    if dim == 2:
        return grads[0][...,1] - grads[1][...,0]

def velocity_field(qdata, dim = 2):
    vfield = qdata[...,1:1+dim] / (qdata[...,0:1] + 1e-7)
    return vfield

def density_plot(p):
    plot = plt.matshow(p, cmap='Spectral_r', vmin=0)
    return plot

def vorticity_plot(x, y, Fx, Fy):
    c = curl(x, y, Fx, Fy)
    v = np.abs(c).max()
    plot = plt.matshow(c, cmap='Spectral_r', vmin=-v, vmax=v)
    return plot

def main():

    file_dir = 'visuals/2d_preds/'
    filenames = os.listdir(file_dir)
    filenames = [file_dir + f for f in filenames]

    arrays = []
    axis = []
    labels = []
    for fname in filenames:
        arr = np.squeeze(np.load(fname))
        if 'GT' in fname:
            axis.append(0)
            labels.append('Ground Truth')
            vmin, vmax = arr[...,0].min(), arr[...,0].max()
            # vmin, vmax = 0, arr[...,0].max()
            t0, arr = arr[0], arr[1:]
        elif 'unscaled' in fname:
            axis.append(2)
            labels.append('Unscaled')
        elif 'ke' in fname or 'allscaled' in fname:
            axis.append(3)
            labels.append('Energy-Scaled')
        else:
            axis.append(1)
            labels.append('Scaled')
        arrays.append(arr)
        
    fig, axes = plt.subplots(len(axis), 6, figsize = (18, 12), layout='constrained',
                            gridspec_kw=dict(width_ratios=[6,1,6,6,6,6], hspace=0, wspace=0.1))
    times = [1, 2, 5, 10]

    print(vmin, vmax)

    for i in range(len(axis)):
        axes[i, 0].imshow(t0[...,0], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

    axes[0, 0].set_title(f"T=0", fontsize=24)

    for i, t in enumerate(times):
        for j, label, arr in zip(axis, labels, arrays):
            ax = axes[j, i+2]
            print(j, arr[t-1,...,0].sum())
            ax.imshow(arr[t-1,...,0], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

            if j == 0:
                ax.set_title(f'T={t}', fontsize=24)

            if i == 0:
                ax.set_ylabel(label, fontsize=24)
                axes[j, i+1].remove()
    
    for i in range(1, len(axis)):
        axes[i, 0].axis('off')
    
    fig.suptitle('Flow Density Progression for 2D CFD', fontsize=28)
    # plt.subplots_adjust(wspace=0.1, hspace=-0.25)
    plt.savefig('visuals/2D_flow_progression_allscaled.png', bbox_inches='tight')
    #plt.show()

def main2():
    file_dir = 'out/output_for_visuals/2D/'
    filenames = os.listdir(file_dir)
    filenames = [file_dir + f for f in filenames]
    filenames.append('out/output_for_visuals/GT_testing_2D.npy')
    
    mods = ['OneStep', 'AutoRegressive']

    axis_map = {'Ground Truth': 0,
                'FNO (unscaled)': 1,
                'FNO (scaled)': 2,
                'PI-FNO (unscaled)': 3,
                'PI-FNO (scaled)': 4,
                'FNO (all-scaled)': 5
            }

    for mod in mods:
        arrays = []
        axis = []
        labels = []
        for fname in filenames:
            arr = np.squeeze(np.load(fname))
            if 'GT' in fname:
                label = 'Ground Truth'
                vmin, vmax = arr[...,0].min(), arr[...,0].max()
                t0, arr = arr[0], arr[1:]
            else:
                dataset, model, lr, lr2, pde_weight, scaling = fname.split('-')
                
                if model != mod:
                    continue

                if pde_weight.endswith('0.0'):
                    label = 'FNO '
                elif pde_weight.endswith('0.01'):
                    label = 'PI-FNO '
                
                if scaling.startswith('un'):
                    label += '(unscaled)'
                elif scaling.startswith('all'):
                    label += '(all-scaled)'
                else:
                    label += '(scaled)'
            
            labels.append(label)
            axis.append(axis_map[label])
            arrays.append(arr)
            
        fig, axes = plt.subplots(len(axis), 6, figsize = (18, 19), layout='constrained',
                                gridspec_kw=dict(width_ratios=[6,1,6,6,6,6], hspace=0, wspace=0.1))
        times = [1, 2, 5, 10]

        for i in range(len(axis)):
            axes[i, 0].imshow(t0[...,0], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

        axes[0, 0].set_title(f"T=0", fontsize=24)

        for i, t in enumerate(times):
            for j, label, arr in zip(axis, labels, arrays):
                ax = axes[j, i+2]
                ax.imshow(arr[t-1,...,0], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])

                if j == 0:
                    ax.set_title(f'T={t}', fontsize=24)

                if i == 0:
                    ax.set_ylabel(label, fontsize=24)
                    axes[j, i+1].remove()
        
        for i in range(len(axis)-1):
            axes[i+1, 0].axis('off')
            
        fig.suptitle(f'Flow Density Progression for 2D CFD ({mod})', fontsize=28)
        # plt.subplots_adjust(wspace=0.1, hspace=-0.25)
        plt.savefig(f'visuals/2D_flow_progression_all_{mod}.png', bbox_inches='tight')
        #plt.show()


if __name__ == '__main__':
    main()
    main2()
