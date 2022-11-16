import numpy as np
import scipy.sparse as sp
import glob
import torch
from torch_geometric.utils import get_laplacian
from .dynamic_graph import DynamicGraph

class PeriodicGraph(DynamicGraph):
    def __init__(self, grid, solution, timesteps=0, use_cuda=False):
        self.grid, self.solution = self.cut_periodic(grid, solution)
        self.timesteps = timesteps
        self.use_cuda = use_cuda
        
        trig_grid = self.periodicize_grid(self.grid)
        self.locations = trig_grid.reshape((-1, trig_grid.shape[-1]))
        self.adj = self.build_adj_matrix()
        
        self.train = self.solution[:int(.8 * len(self.solution))]
        self.test = self.solution[int(.8 * len(self.solution)):]
        self.training = True

    def cut_periodic(self, grid, solution):
        #cut features to ensure periodicity
        xindices = np.arange(grid.shape[0])
        yindices = np.arange(grid.shape[1])
        zindices = np.arange(grid.shape[2])

        helper = lambda comparison, axes: np.argmax(np.sum(comparison, axis=axes)) + 1

        #find periodic cutoffs
        xcutoff = helper(solution[:, 1:] == solution[:, :1, :, :, :], (0, 2, 3, 4))
        ycutoff = helper(solution[:, :, 1:] == solution[:, :, :1, :, :], (0, 1, 3, 4))
        zcutoff = helper(solution[:, :, :, 1:] == solution[:, :, :, :1, :], (0, 1, 2, 4))

        #cut grid and solution
        grid = grid[:xcutoff, :ycutoff, :zcutoff, :]
        solution = solution[:, :xcutoff, :ycutoff, :zcutoff, :]

        return grid, solution

    def periodicize_grid(self, grid):
        #new coordinate system
        #scale invariant, will produce wrap-around coordinates when combined with WaveTranform
        xcoords = np.linspace(0, 2*np.pi, grid.shape[0])
        ycoords = np.linspace(0, 2*np.pi, grid.shape[1])
        zcoords = np.linspace(0, 2*np.pi, grid.shape[2])

        x, y, z = np.meshgrid(xcoords, ycoords, zcoords)
        trig_grid = np.stack([x, y, z], axis = -1)
        return trig_grid

    def build_adj_matrix(self):
        #edgelist should be the same for both source and target
        edges = self.construct_edgelist(*self.grid.shape[:3], periodic=True)
        return get_laplacian(torch.tensor(edges), torch.ones(edges.shape[1]),
                             normalization='sym')
