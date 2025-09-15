import numpy as np
import scipy.sparse as sp
import glob, os, h5py
import math as mt
import torch
from torch.utils.data import Dataset
from skimage.measure import block_reduce

from .dynamic_grid import *

class DebugGrid(DynamicGrid):
    def __init__(self, shape, timesteps=1, history=1, use_cuda=False):
        # make grid and solution
        samples, dims, duration = shape[0], shape[1:-1], shape[-1]
        grid = np.meshgrid(*[np.linspace(0, 2*np.pi, d, endpoint=False) for d in dims], indexing='xy')
        grid = np.stack(grid, axis=-1)
        solution = np.sin(grid.sum(axis=-1) + np.random.rand(samples, *dims)) + 1
        solution = np.stack([np.roll(solution, shift=(i * np.arange(1, len(dims) + 1)),
                                     axis=tuple(range(1, len(dims)))) for i in range(duration)], axis=-1)
        velocities = duration * np.arange(1, len(dims) + 1) * np.array([2*np.pi/d for d in dims])

        velocities = np.stack([np.ones_like(solution) * v for v in velocities], axis=-1)
        solution = np.expand_dims(solution, -1)
        solution = np.concatenate([solution, solution, velocities], axis=-1)
        if samples == 1:
            solution = np.repeat(solution, 2, axis=0)
        super(DebugGrid, self).__init__(grid, solution, timesteps=timesteps, history=history, use_cuda=use_cuda)

class DebugStaticGrid(DebugGrid):
    def __init__(self, shape, timesteps=1, history=1, use_cuda=False):
        super(DebugStaticGrid, self).__init__(shape, timesteps=timesteps, history=history, use_cuda=use_cuda)

    def sample_grid(self, flow_features, index, num_batches, timesteps=1, hist=1):
        features, targets = super().sample_grid(flow_features, index, num_batches, timesteps=timesteps, hist=hist)
        features = features.reshape(features.shape[:-2] + (1, -1))
        features = features.tile((timesteps, 1))
        return features, targets

class DebugNumpyGrid(DynamicGrid):
    def __init__(self, file, timesteps=1, history=1, use_cuda=False, reduced_resolution=1):
        solution = np.load(file)[0:1] # grab first sample
        solution = np.moveaxis(solution, 1, -2)
        # temp fix for ordering of channels: REMOVE LATER
        # solution = solution[..., [0, -1, 1, 2]] # rho
        solution = np.repeat(solution, 2, axis=0)
        print(solution.shape)
        if reduced_resolution > 1:
            solution = block_reduce(solution, block_size=(1, reduced_resolution, reduced_resolution, 1, 1), func=np.mean)
        grid = np.meshgrid(*[np.linspace(0, 1, d, endpoint=False) for d in solution.shape[1:-2]], indexing='xy')
        grid = np.stack(grid, axis=-1)
        super(DebugNumpyGrid, self).__init__(grid, solution, timesteps=timesteps, history=history, use_cuda=use_cuda)
        