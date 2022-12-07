import numpy as np
import scipy.sparse as sp
import glob
import torch
from torch_geometric.utils import get_laplacian

from .dynamic_graph import DynamicGraph
from .periodic_graph import PeriodicGraph
from .minibatch_graph import MiniBatchGraph

class TGVGraph(DynamicGraph):
    def __init__(self, grd_path, solution_path, timesteps=1, use_cuda=False):
        if not grd_path.endswith('.npy') and not solution_path.endswith('.npy'):
            self.grid, self.solution = self.read_input_directory(grd_path, solution_path)
            np.save(grd_path.replace('.', '_') + '.npy', self.grid)
            np.save(solution_path + '_sol.npy', self.solution)
        else:
            self.grid, self.solution = np.load(grd_path), np.load(solution_path)
        super(TGVGraph, self).__init__(self.grid, self.solution, timesteps=timesteps, use_cuda=use_cuda)

    def read_input_directory(self, grd_path, solution_path):
        locations = self.read_grd_data(grd_path)

        flow_features = []
        for i in range(1001):
            if i != 0:
                i *= 100
            filename = solution_path + f'{i}'
            flow_features.append(self.read_solution_data(filename))
        features = np.stack(flow_features, axis = 0)

        return locations, features

    def read_grd_data(self, fname):
        data = np.fromfile(fname, dtype='>i4')
        ni, nj, nk = data[1:4]
        x = data[6:-1].view(dtype='>f4')
        # (y, x, channels)
        x = x.reshape((3, nk, nj, ni)).transpose()
        return x

    def read_solution_data(self, fname):
        data = np.fromfile(fname, dtype='>i4')
        ni, nj, nk = data[1:4]
        q = data[7:-1].view(dtype='>f4')
        # (y, x, channels)
        # channels are d, vx, vy, vz, E
        q = q.reshape((5, nk, nj, ni)).transpose()
        return q

class PeriodicTGVGraph(PeriodicGraph):
    def __init__(self, grd_path, solution_path, timesteps=1, use_cuda=False):
        if not grd_path.endswith('.npy') and not solution_path.endswith('.npy'):
            self.grid, self.solution = TGVGraph.read_input_directory(self, grd_path, solution_path)
            np.save(grd_path.replace('.', '_') + '.npy', self.grid)
            np.save(solution_path + '_sol.npy', self.solution)
        else:
            self.grid, self.solution = np.load(grd_path), np.load(solution_path)
        self.grid, self.solution = self.cut_periodic(self.grid, self.solution)
        super(PeriodicTGVGraph, self).__init__(self.grid, self.solution, timesteps=timesteps, use_cuda=use_cuda)

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

class MiniBatchTGVGraph(MiniBatchGraph):
    def __init__(self, grd_path, solution_path, width, depth, timesteps=1, use_cuda=False):
        if not grd_path.endswith('.npy') and not solution_path.endswith('.npy'):
            self.grid, self.solution = TGVGraph.read_input_directory(self, grd_path, solution_path)
            np.save(grd_path.replace('.', '_') + '.npy', self.grid)
            np.save(solution_path + '_sol.npy', self.solution)
        else:
            self.grid, self.solution = np.load(grd_path), np.load(solution_path)
        self.grid, self.solution = PeriodicTGVGraph.cut_periodic(self, self.grid, self.solution)
        super(MiniBatchTGVGraph, self).__init__(self.grid, self.solution, width, depth,
                                                timesteps=timesteps, use_cuda=use_cuda)
