import numpy as np
import scipy.sparse as sp
import glob
import torch
from torch_geometric.utils import get_laplacian, coalesce
from torch.utils.data import Dataset

from .pdebench_graph import PDEBenchGraph
from .tgv_graph import TGVGraph

# MINIBATCH EVERYTHING with collate_fn

class DynamicGrid(Dataset):
    def __init__(self, grid, solution, num_channels, timesteps=1, history=0, use_cuda=False):
        self.grid = grid
        self.solution = solution
        self.num_channels = num_channels
        self.timesteps = timesteps
        self.history = history
        self.use_cuda = use_cuda
        self.nfeats = self.num_channels
        self.ndims = self.grid.shape[-1]
        
        self.train = self.solution[..., :int(.9 * self.solution.shape[-1])]
        self.test = self.solution[..., int(.9 * self.solution.shape[-1]):]
        self.num_batches = self.train[0].shape[-1] // self.num_channels
        self.is_training = True
    
    def sample_grid(self, flow_features, index, tdif=1, hist=0):
        t = index // self.num_batches
        c = (index % self.num_batches) * self.num_channels
        features = flow_features[t: t+hist][..., c: c + self.num_channels]
        targets = flow_features[t+hist+1: t+hist+tdif+1][..., c: c + self.num_channels]

        #shape into feature matrics
        features = np.moveaxis(features, 0, -2).reshape(features.shape[:-2] + (-1,))

        features = torch.from_numpy(features.astype(np.float32)).requires_grad_()
        targets = torch.from_numpy(targets.astype(np.float32)).requires_grad_()
        locations = torch.from_numpy(self.grid.astype(np.float32)).requires_grad_()

        mask = None
        adj = None

        if self.use_cuda:
            locations = locations.to(device='cuda')
            features = features.to(device='cuda')
            targets = targets.to(device='cuda')


        return adj, locations, features, targets, mask

    def training(self):
        self.is_training = True

    def testing(self):
        self.is_training = False

    def __getitem__(self, index):
        if self.is_training:
            return self.sample_grid(self.train, index, tdif=self.timesteps, hist=self.history)
        else:
            return self.sample_grid(self.test, index, tdif=self.timesteps, hist=self.history)

    def __len__(self):
        if self.is_training:
            return (len(self.train) - self.timesteps - self.history) * self.num_batches
        else:
            return (len(self.test) - self.timesteps - self.history) * self.num_batches


class PDEBenchGrid(DynamicGrid):
    def __init__(self, *args, timesteps=1, history=0, use_cuda=False, **kwargs):
        grid, solution, num_channels = PDEBenchGraph.read_input_directory(self, *args, **kwargs)
        super(PDEBenchGrid, self).__init__(grid, solution, num_channels,
                                           timesteps=timesteps, history=history, use_cuda=use_cuda)

class TGVGrid(DynamicGrid):
    def __init__(self, grd_path, solution_path, timesteps=1, history=0, use_cuda=False):
        if not grd_path.endswith('.npy') and not solution_path.endswith('.npy'):
            self.grid, self.solution = TGVGraph.read_input_directory(self, grd_path, solution_path)
            np.save(grd_path.replace('.', '_') + '.npy', self.grid)
            np.save(solution_path + '_sol.npy', self.solution)
        else:
            self.grid, self.solution = np.load(grd_path), np.load(solution_path)
        self.grid, self.solution = PeriodicTGVGraph.cut_periodic(self, self.grid, self.solution)
        super(DynamicGrid, self).__init__(self.grid, self.solution, 5,
                                          timesteps=timesteps, history=history, use_cuda=use_cuda)
