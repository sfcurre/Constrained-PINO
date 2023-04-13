import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import argparse

import torch

from source.data.dynamic_grid import PDEBenchGrid
from source.losses.navier_stokes_loss import pde_CFD2d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--run-true', action='store_true')
    return parser.parse_args()

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

def animate_channels(mat, name):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    vmax = mat.max(axis=(0, 1, 2))
    vmin = mat.min(axis=(0, 1, 2))

    def update(n=0):
        pls = []
        for i in range(mat.shape[-1]):
            pl = axes[i].imshow(mat[n, ..., i], cmap='jet', animated=True, vmin=vmin[i], vmax=vmax[i], origin='lower')
            if n == 0:
                fig.colorbar(pl, ax=axes[i], location='bottom')
            
            pls.append(pl)
            #pl = ax.imshow(p_mat[n], cmap='jet', animated=True, origin='lower')
        return pls

    fig.suptitle('Fluid Channels (density, vx, vy, pressure)')
    anim = animation.ArtistAnimation(fig, [update(n) for n in tqdm(range(len(mat)))], blit=True, interval=100)
    print('Saving mp4...')
    anim.save(f'out/visuals/fluid_density_{name}.mp4', writer='ffmpeg')


def animate_density(p_mat, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    vmax = p_mat.max()
    vmin = p_mat.min()

    def update(n=0):
        pl = ax.imshow(p_mat[n], cmap='jet', animated=True, vmin=vmin, vmax=vmax, origin='lower')
        #pl = ax.imshow(p_mat[n], cmap='jet', animated=True, origin='lower')
        return pl,

    plt.title('Fluid Density')
    anim = animation.ArtistAnimation(fig, [update(n) for n in tqdm(range(len(p_mat)))], blit=True, interval=100)
    print('Saving mp4...')
    anim.save(f'out/visuals/fluid_density_{name}.mp4', writer='ffmpeg')


def animate_vorticity(velocities, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    curls = np.array([curl(velocities[n]) for n in tqdm(range(len(velocities)))])
    v = np.abs(curls).max()
    print(curls.shape, v)

    def update(n=0):
        c = curls[n]
        pl = ax.imshow(c, cmap='bwr', vmin=-v, vmax=v, origin='lower')
        return pl,

    plt.title('Fluid Vorticity')
    anim = animation.ArtistAnimation(fig, [update(n) for n in range(len(velocities))], blit=True, interval=100)
    print('Saving mp4...')
    anim.save(f'out/visuals/fluid_vorticity_{name}.mp4', writer='ffmpeg')


def main():
    args = parse_args()

    if args.run_true:
        filename = "/fs/scratch/PAS2170/pdebench/data/2D/CFD/2D_Train_Rand/2d_cfd_rand_m0.1_eta1e-08_zeta1e-08_periodic_512_train.hdf5"
        timesteps = 100
        data = PDEBenchGrid(filename, timesteps=timesteps, history=1, use_cuda=False, reduced_batch=2, reduced_resolution=2)
        print(len(data.solution))        

        for i in range(2):
            if i == 0:
                data.training()
            else:
                data.testing()
            adj, locations, features, targets, mask = data[0]

            features = features.detach().numpy()
            targets = [t.detach().numpy() for t in targets]
            solution = np.concatenate([features[None, ...], targets], axis = 0)
            print(solution.shape)
            solution = solution.reshape((len(solution),) + data.grid.shape[:-1] + (data.num_channels,))
            solution = solution[..., 0, :]

            animate_channels(np.tile(solution, (1, 1, 1)), f'T2D0_{"training" if i==0 else "testing"}')

            total_density = np.sum(solution[..., 0], axis = (1, 2))
            squared_vx = solution[..., 1] ** 2
            squared_vy = solution[..., 2] ** 2
            total_pressure = np.sum(solution[..., 3], axis = (1, 2))
            total_kinetic = np.sum(0.5 * solution[..., 0] * (squared_vx + squared_vy), axis = (1, 2))
            total_energy = total_kinetic + total_pressure

            change_density = total_density[1:] - total_density[:-1]
            change_pressure = total_pressure[1:] - total_pressure[:-1]
            change_kinetic = total_kinetic[1:] - total_kinetic[:-1]
            change_energy = total_energy[1:] - total_energy[:-1]

            fig = plt.figure(figsize=(8,8))
            x = np.arange(len(total_density))
            plt.plot(x, total_density, label='Density')
            plt.plot(x, total_pressure, label='Pressure')
            plt.plot(x, total_kinetic, label='Kinetic')
            plt.plot(x, total_energy, label='Energy')
            plt.legend()
            plt.xlabel('t')
            plt.ylabel('Total')
            plt.savefig(f'out/visuals/total_{"training" if i==0 else "testing"}.png')
            del fig

    full_out = np.load(f'out/predictions/{args.name}_full.npy').reshape((30, 256, 256, 1, -1))
    test_out = np.load(f'out/predictions/{args.name}_test.npy').reshape((30, 256, 256, 1, -1))

    animate_channels(full_out[..., 0, :], f'{args.name}')
    animate_channels(test_out[..., 0, :], f'{args.name}_test')


if __name__ == '__main__':
    main()
