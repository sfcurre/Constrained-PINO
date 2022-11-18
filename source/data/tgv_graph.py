import numpy as np
import scipy.sparse as sp
import glob
import torch
from torch_geometric.utils import get_laplacian

from .dynamic_graph import DynamicGraph
from .periodic_graph import PeriodicGraph

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
            self.grid, self.solution = self.read_input_directory(grd_path, solution_path)
            np.save(grd_path.replace('.', '_') + '.npy', self.grid)
            np.save(solution_path + '_sol.npy', self.solution)
        else:
            self.grid, self.solution = np.load(grd_path), np.load(solution_path)
        super(PeriodicTGVGraph, self).__init__(self.grid, self.solution, timesteps=timesteps, use_cuda=use_cuda)

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

