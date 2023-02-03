import numpy as np
import scipy.sparse as sp
import glob
import torch
from torch_geometric.utils import get_laplacian
from .periodic_graph import PeriodicGraph

class MiniBatchGraph(PeriodicGraph):
    def __init__(self, grid, solution, width, depth, num_channels, timesteps=1, use_cuda=False):
        self.grid = grid
        self.solution = solution
        self.width = width
        self.depth = depth
        self.num_channels = num_channels
        self.timesteps = timesteps
        self.use_cuda = use_cuda
        self.nfeats = self.solution.shape[-1]
        self.ndims = self.grid.shape[-1]
        
        self.trig_grid = self.periodicize_grid(self.grid)
        self.locations = self.trig_grid.reshape((-1, self.trig_grid.shape[-1]))
        self.mini_adj = self.build_minibatch_adj_matrix()
        self.adj = self.build_adj_matrix()
        
        self.train = self.solution[:int(.8 * len(self.solution))]
        self.test = self.solution[int(.8 * len(self.solution)):]
        self.num_batches = self.train[0].shape[-1] // self.num_channels
        self.super_batch_shape = self.get_super_batch_shape()
        self.is_training = True

    def build_minibatch_adj_matrix(self):
        #edgelist should be the same for both source and target
        window = self.width + 2 * self.depth + (self.timesteps - 1)
        edges = self.construct_edgelist(window, window, window, periodic=False)
        return get_laplacian(torch.tensor(edges), torch.ones(edges.shape[1]),
                             normalization='sym')
        
    def sample_minibatch(self, flow_features, t, x, y, z, c, tdif=1):
        features, targets = flow_features[t], flow_features[t+1: t+tdif+1]

        # roll features, targets, locations
        xshift, yshift, zshift = self.depth - np.array([x, y, z])
        features = np.roll(features, (xshift, yshift, zshift), axis=(0, 1, 2))
        targets = np.roll(targets, (xshift, yshift, zshift), axis=(1, 2, 3))
        locations = np.roll(self.trig_grid, (xshift, yshift, zshift), axis=(0, 1, 2))

        #select window
        size = self.width + 2 * self.depth + (self.timesteps - 1)
        features = features[:size, :size, :size, c: c + self.num_channels]
        targets = targets[:, :size, :size, :size, c: c + self.num_channels]
        locations = locations[:size, :size, :size, c: c + self.num_channels]
        
        mask = np.ones(locations.shape[:-1])
        mask[self.depth:self.depth + self.width,
             self.depth:self.depth + self.width,
             self.depth:self.depth + self.width] = 0

        #shape into feature matrices
        features = features.reshape((-1, features.shape[-1]))
        targets = targets.reshape((targets.shape[0], -1, targets.shape[-1]))
        locations = locations.reshape((-1, locations.shape[-1]))
        mask = mask.reshape((-1, 1))

        features = torch.from_numpy(features.astype(np.float32)).requires_grad_()
        targets = torch.from_numpy(targets.astype(np.float32)).requires_grad_()
        locations = torch.from_numpy(locations.astype(np.float32)).requires_grad_()
        mask = torch.from_numpy(mask.astype(np.float32)).requires_grad_()
        
        adj = self.mini_adj
        if self.use_cuda:
            adj = (adj[0].cuda(), adj[1].cuda())
            locations = locations.cuda()
            features = features.cuda()
            targets = targets.cuda()
            mask = mask.cuda()

        return adj, locations, features, targets, mask

    def __getitem__(self, index):
        if self.is_training:
            t, x, y, z, c = np.unravel_index(index, shape=self.super_batch_shape)
            x, y, z = x * self.width, y *self.width, z * self.width
            return self.sample_minibatch(self.train, t, x, y, z, c, tdif=self.timesteps)
        else:
            return self.sample_graph(self.test, index, tdif=self.timesteps)

    def __len__(self):
        if self.is_training:
            return np.prod(self.super_batch_shape, dtype=int)
        else:
            return (len(self.test) - self.timesteps) * self.num_batches

    def get_super_batch_shape(self):
        shape = [len(self.train) - self.timesteps]
        for dim in range(3):
            shape.append(self.train[0].shape[dim] // self.width + 
                         bool(self.train[0].shape[dim] % self.width))
        shape.append(self.num_batches)
        return shape