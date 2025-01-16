<<<<<<< Updated upstream
import numpy as np
import scipy.sparse as sp
import glob, os, h5py
import math as mt
import torch
from torch_geometric.utils import get_laplacian, coalesce
from torch.utils.data import Dataset

from .dynamic_grid import *

class StaticGrid(DynamicGrid):
    def __init__(self, grid, solution, timesteps=1, use_cuda=False):
        self.grid = grid
        # batch, x, y, z, t, ch
        self.solution = solution
        self.duration = self.solution.shape[-2]
        self.timesteps = min(timesteps, self.duration - 1)
        self.history = 1
        self.use_cuda = use_cuda
        self.num_channels = self.solution.shape[-1]
        self.nfeats = self.num_channels
        self.ndims = self.grid.shape[-1]
        
        self.train = self.solution[:int(.9 * self.solution.shape[0])]
        self.test = self.solution[int(.9 * self.solution.shape[0]):]
        self.num_batches = self.train.shape[0]
        self.num_test_batches = self.test.shape[0]
        self.is_training = True
    
    def sample_grid(self, flow_features, index, num_batches, timesteps=1):
        b = index % num_batches
        t = index // num_batches
        features = flow_features[b:b+1][..., t:t+1, :]
        targets = flow_features[b:b+1][..., t+1:t+timesteps+1, :]

        #shape into feature matrics
        features = torch.from_numpy(features.astype(np.float32)).requires_grad_()
        targets = torch.from_numpy(targets.astype(np.float32)).requires_grad_()
        locations = torch.from_numpy(self.grid.astype(np.float32)).requires_grad_()

        features = features.tile((timesteps, 1))
        locations = locations.unsqueeze(0).unsqueeze(-2)
        locations = locations.tile((timesteps, 1))
        if self.use_cuda:
            locations = locations.to(device='cuda')
            features = features.to(device='cuda')
            targets = targets.to(device='cuda')

        return locations, features, targets

    def training(self):
        self.is_training = True

    def testing(self):
        self.is_training = False

    def __getitem__(self, index):
        if self.is_training:
            return self.sample_grid(self.train, index, self.num_batches, timesteps=self.timesteps)
        else:
            return self.sample_grid(self.test, index, self.num_test_batches, timesteps=self.timesteps)

    def __len__(self):
        if self.is_training:
            return (self.solution.shape[-2] - self.timesteps) * self.num_batches
        else:
            return (self.solution.shape[-2] - self.timesteps) * self.num_test_batches


class PDEBenchStaticGrid(StaticGrid, PDEBenchGrid):
    def __init__(self, *args, timesteps=1, use_cuda=False, **kwargs):
        grid, solution = self.read_input_directory(*args, **kwargs)
        super(PDEBenchStaticGrid, self).__init__(grid, solution, timesteps=timesteps, use_cuda=use_cuda)
        
class PINOStaticGrid(StaticGrid, PINOGrid):
    def __init__(self, *args, timesteps=1, use_cuda=False, **kwargs):
        grid, solution  = self.read_input_directory(*args, **kwargs)
        super(PINOGrid, self).__init__(grid, solution, timesteps=timesteps, use_cuda=use_cuda)
=======
import numpy as np
import scipy.sparse as sp
import glob, os, h5py
import math as mt
import torch
from torch_geometric.utils import get_laplacian, coalesce
from torch.utils.data import Dataset

from .dynamic_grid import *

class StaticGrid(DynamicGrid):
    def __init__(self, grid, solution, timesteps=1, use_cuda=False):
        self.grid = grid
        # batch, x, y, z, t, ch
        self.solution = solution
        self.timesteps = timesteps
        self.history = 1
        self.use_cuda = use_cuda
        self.num_channels = self.solution.shape[-1]
        self.nfeats = self.num_channels
        self.ndims = self.grid.shape[-1]
        self.duration = self.solution.shape[-2]
        
        self.train = self.solution[:int(.9 * self.solution.shape[0])]
        self.test = self.solution[int(.9 * self.solution.shape[0]):]
        self.num_batches = self.train.shape[0]
        self.num_test_batches = self.test.shape[0]
        self.is_training = True
    
    def sample_grid(self, flow_features, index, num_batches, timesteps=1):
        b = index % num_batches
        t = index // num_batches
        features = flow_features[b:b+1][..., t:t+1, :]
        targets = flow_features[b:b+1][..., t+1:t+timesteps+1, :]

        #shape into feature matrics
        features = torch.from_numpy(features.astype(np.float32)).requires_grad_()
        targets = torch.from_numpy(targets.astype(np.float32)).requires_grad_()
        locations = torch.from_numpy(self.grid.astype(np.float32)).requires_grad_()

        features = features.tile((timesteps, 1))
        locations = locations.unsqueeze(0).unsqueeze(-2)
        locations = locations.tile((timesteps, 1))
        if self.use_cuda:
            locations = locations.to(device='cuda')
            features = features.to(device='cuda')
            targets = targets.to(device='cuda')

        return locations, features, targets

    def training(self):
        self.is_training = True

    def testing(self):
        self.is_training = False

    def __getitem__(self, index):
        if self.is_training:
            return self.sample_grid(self.train, index, self.num_batches, timesteps=self.timesteps)
        else:
            return self.sample_grid(self.test, index, self.num_test_batches, timesteps=self.timesteps)

    def __len__(self):
        if self.is_training:
            return (self.solution.shape[-2] - self.timesteps) * self.num_batches
        else:
            return (self.solution.shape[-2] - self.timesteps) * self.num_test_batches


class PDEBenchStaticGrid(StaticGrid, PDEBenchGrid):
    def __init__(self, *args, timesteps=1, use_cuda=False, **kwargs):
        grid, solution = self.read_input_directory(*args, **kwargs)
        super(PDEBenchStaticGrid, self).__init__(grid, solution, timesteps=timesteps, use_cuda=use_cuda)
        
class PINOStaticGrid(StaticGrid, PINOGrid):
    def __init__(self, *args, timesteps=1, use_cuda=False, **kwargs):
        grid, solution  = self.read_input_directory(*args, **kwargs)
        super(PINOGrid, self).__init__(grid, solution, timesteps=timesteps, use_cuda=use_cuda)
>>>>>>> Stashed changes
