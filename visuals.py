import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import argparse

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
        true_data = np.load('TGV/ijk_snapshots/ijkmvmv_sol.npy')[:, :63, :63, :63]
        
        i = 16
        true_xslice = true_data[:, i, ...]
        true_yslice = true_data[:, :, i, ...]
        true_zslice = true_data[:, :, :, i, ...]

        animate_density(np.tile(true_xslice[..., 0], (1, 1, 1)), f'Tx{i}')
        animate_density(np.tile(true_yslice[..., 0], (1, 1, 1)), f'Ty{i}')
        animate_density(np.tile(true_zslice[..., 0], (1, 1, 1)), f'Tz{i}')
    
    full_out = np.load(f'out/predictions/{args.name}_full.npy').reshape((1000, 63, 63, 63, -1))
    test_out = np.load(f'out/predictions/{args.name}_test.npy').reshape((200, 63, 63, 63, -1))

    i = 16
    full_xslice = full_out[:, i, ...]
    full_yslice = full_out[:, :, i, ...]
    full_zslice = full_out[:, :, :, i, ...]

    animate_density(full_xslice[..., 0], f'{args.name}_x16')
    animate_density(full_yslice[..., 0], f'{args.name}_y16')
    animate_density(full_zslice[..., 0], f'{args.name}_z16')

if __name__ == '__main__':
    main()
