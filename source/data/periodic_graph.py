import numpy as np
import scipy.sparse as sp
import glob
import torch
from torch_geometric.utils import get_laplacian
from .dynamic_graph import DynamicGraph

class PeriodicGraph(DynamicGraph):
    def __init__(self, grid, solution, num_channels, timesteps=1, use_cuda=False):
        self.grid = grid
        self.solution = solution
        self.num_channels = num_channels
        self.timesteps = timesteps
        self.use_cuda = use_cuda
        self.nfeats = self.solution.shape[-1]
        self.ndims = self.grid.shape[-1]
        
        trig_grid = self.periodicize_grid(self.grid)
        self.locations = trig_grid.reshape((-1, trig_grid.shape[-1]))
        self.adj = self.build_adj_matrix()
        
        self.train = self.solution[..., :int(.9 * self.solution.shape[-1])]
        self.test = self.solution[..., int(.9 * self.solution.shape[-1]):]
        self.num_batches = self.train[0].shape[-1] // self.num_channels
        self.is_training = True

    def periodicize_grid(self, grid):
        #new coordinate system
        #tested, checks out
        #scale invariant, will produce wrap-around coordinates when combined with WaveTranform
        
        # FIX SCALING
        xcoords = np.linspace(0, 2*np.pi, grid.shape[0] + 1)[:-1]
        ycoords = np.linspace(0, 2*np.pi, grid.shape[1] + 1)[:-1]
        zcoords = np.linspace(0, 2*np.pi, grid.shape[2] + 1)[:-1]

        x, y, z = np.meshgrid(xcoords, ycoords, zcoords, indexing='ij')
        trig_grid = np.stack([x, y, z], axis = -1)
        return trig_grid

    def build_adj_matrix(self):
        #edgelist should be the same for both source and target
        edges = self.construct_edgelist(*self.grid.shape[:3], periodic=True)
        return get_laplacian(torch.tensor(edges), torch.ones(edges.shape[1]),
                             normalization='sym')
