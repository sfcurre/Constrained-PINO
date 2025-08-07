import numpy as np
import scipy.sparse as sp
import glob, os, h5py
import math as mt
import torch
from torch.utils.data import Dataset

from .dynamic_grid import *

class DebugGrid(DynamicGrid):
    def __init__(self, shape, timesteps=1, history=1, use_cuda=False):
        # make grid and solution
        samples, dims, duration = shape[0], shape[1:-1], shape[-1]
        grid = np.meshgrid(*[np.linspace(0, 1, d) for d in dims], indexing='ij')
        grid = np.stack(grid, axis=-1)
        solution = np.sin(grid.sum(axis=-1)*(2*np.pi + np.random.rand(samples, *grid.shape[:-1]))) + 1
        solution = np.stack([np.roll(solution, shift=np.arange(1, len(dims)) + i,
                                     axis=tuple(range(1, len(dims)))) for i in range(duration)], axis=-1)
        velocities = np.arange(1, len(dims)) * np.array([1/d for d in dims])
        velocities = np.stack([np.ones_like(solution) * v for v in velocities], axis=-1)
        solution = np.expand_dims(solution, -1)
        solution = np.concatenate([solution, velocities, solution], axis=-1)
        super(DebugGrid, self).__init__(grid, solution, timesteps=timesteps, history=history, use_cuda=use_cuda)

class DebugStaticGrid(DebugGrid):
    def __init__(self, shape, timesteps=1, history=1, use_cuda=False):
        super(DebugStaticGrid, self).__init__(shape, timesteps=timesteps, history=history, use_cuda=use_cuda)

    def sample_grid(self, flow_features, index, num_batches, timesteps=1, hist=1):
        features, targets = super().sample_grid(flow_features, index, num_batches, timesteps=timesteps, hist=hist)
        features = features.reshape(features.shape[:-2] + (1, -1))
        features = features.tile((timesteps, 1))
        return features, targets
