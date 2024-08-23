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
    parser.add_argument('--run-true', type=str, default='')
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

def animate_channels(mat, name, vmin=None, vmax=None):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    if vmax is None:
        vmax = mat.max(axis=(0, 1, 2))
    if vmin is None:
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
    anim.save(f'out/visuals/{name}_channels.mp4', writer='ffmpeg')


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
    anim.save(f'out/visuals/{name}_density.mp4', writer='ffmpeg')


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
    anim.save(f'out/visuals/{name}_vorticity.mp4', writer='ffmpeg')

def plot_totals(solution, name):
    while len(solution.shape) < 5:
        solution = np.expand_dims(solution, axis=-2)

    total_density = np.sum(solution[..., 0], axis = (1, 2, 3))
    squared_velocity_sum = np.sum(solution[..., 1:-1] ** 2, axis = -1)
    total_vx = np.sum(solution[..., 1], axis=(1, 2, 3))
    total_vy = np.sum(solution[..., 2], axis=(1, 2, 3))
    total_pressure = np.sum(solution[..., -1], axis = (1, 2, 3))
    total_kinetic = np.sum(0.5 * solution[..., 0] * (squared_velocity_sum), axis = (1, 2, 3))
    total_energy = total_kinetic + 1.5 * total_pressure

    print(name)
    print(total_density)
    print(total_pressure)
    print(total_vx)
    print(total_vy)
    print(total_kinetic)

    fig = plt.figure(figsize=(8,8))
    x = np.arange(len(total_density))
    plt.plot(x, total_density, label='Density')
    plt.plot(x, total_pressure, label='Pressure')
    plt.plot(x, total_vx, label='Vx')
    plt.plot(x, total_vy, label='Vy')
    plt.plot(x, squared_velocity_sum.sum(axis=(1, 2, 3)), label='V2')
    plt.plot(x, total_kinetic, label='Kinetic')
    plt.plot(x, total_energy, label='Energy')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Total')
    plt.savefig(f'out/visuals/{name}_totals.png')
    del fig

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
    args = parse_args()

    vmin, vmax = None, None

    if args.run_true:
        timesteps = 100
        data = PDEBenchGrid(args.run_true, timesteps=timesteps, history=1, use_cuda=False, reduced_batch=8, reduced_resolution=1)
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
            solution_all = solution.reshape((len(solution),) + data.grid.shape[:-1] + (data.num_channels,))
            solution = solution_all[10:]

            name = f'GT_{"training" if i==0 else "testing"}'


            if '2D' in args.run_true:
                name = '2D/' + name
                animate_channels(solution[..., 0, :], name)
            elif '1D' in args.run_true:
                name = '1D/' + name
                plot_1d(solution, name)
            else: #3D
                name = '3D/' + name
            
            np.save(f'data/{name}.npy', solution_all[9:])
            plot_totals(solution, name)
        
            if i == 0:
                vmin = np.min(solution, axis=tuple(range(len(solution.shape) - 1)))
                vmax = np.max(solution, axis=tuple(range(len(solution.shape) - 1)))
            else:
                vmin_test = np.min(solution, axis=tuple(range(len(solution.shape) - 1)))
                vmax_test = np.max(solution, axis=tuple(range(len(solution.shape) - 1)))

    if '2D' in args.name:
        full_out = np.load(f'out/predictions/{args.name}_full.npy').reshape((11, 256, 256, 1, -1))
        test_out = np.load(f'out/predictions/{args.name}_test.npy').reshape((11, 256, 256, 1, -1))

        animate_channels(full_out[..., 0, :], f'2D/{args.name}', vmin=vmin, vmax=vmax)
        animate_channels(test_out[..., 0, :], f'2D/{args.name}_test', vmin=vmin_test, vmax=vmax_test)

        plot_totals(full_out, f'2D/{args.name}')
        plot_totals(test_out, f'2D/{args.name}_test')

    elif '1D' in args.name:
        full_out = np.load(f'out/predictions/{args.name}_full.npy').reshape((91, 512, 1, 1, -1))
        test_out = np.load(f'out/predictions/{args.name}_test.npy').reshape((91, 512, 1, 1, -1))

        plot_1d(full_out, f'1D/{args.name}', vmin=vmin, vmax=vmax)
        plot_1d(test_out, f'1D/{args.name}_test', vmin=vmin_test, vmax=vmax_test)
        
        plot_totals(full_out, f'1D/{args.name}')
        plot_totals(test_out, f'1D/{args.name}_test')

    else: #3D
        full_out = np.load(f'out/predictions/{args.name}_full.npy').reshape((11, 64, 64, 64, -1))
        test_out = np.load(f'out/predictions/{args.name}_test.npy').reshape((11, 64, 64, 64, -1))

        plot_totals(full_out, f'3D/{args.name}')
        plot_totals(test_out, f'3D/{args.name}_test')

        mag_velocity = np.sum(test_out[..., 1:-1]**2, axis=(1, 2, 3, 4))
        np.save(f'out/predictions/{args.name}_total_velocity.npy', mag_velocity)


if __name__ == '__main__':
    main()
