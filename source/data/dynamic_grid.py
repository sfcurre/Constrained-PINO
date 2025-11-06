import numpy as np
import scipy.sparse as sp
import glob, os, h5py
import math as mt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from ..utils import convert_to_conservative_form

class DynamicGrid(Dataset):
    def __init__(self, grid, solution, timesteps=1, history=1, use_cuda=False):
        self.grid = grid
        # batch, x, y, z, t, ch
        self.solution = solution
        print('Solution shape:', self.solution.shape)
        print('Grid shape:', self.grid.shape)
        self.duration = self.solution.shape[-2]
        self.spacing = [np.diff(self.grid[...,i], axis=self.grid.shape[-1] - (i+1)).mean() for i in range(self.grid.shape[-1])]
        self.spacing.append(1 / (self.duration-1))
        print('Grid spacing:', self.spacing)
        
        self.timesteps = min(timesteps, self.duration - history)
        self.history = history
        self.use_cuda = use_cuda
        self.num_channels = self.solution.shape[-1]
        self.nfeats = self.num_channels
        self.ndims = self.grid.shape[-1]
        self.start_index = 0

        self.train = self.solution[:int(.8 * self.solution.shape[0])]
        self.test = self.solution[int(.8 * self.solution.shape[0]):int(.9 * self.solution.shape[0])]
        self.val = self.solution[int(.9 * self.solution.shape[0]):]
        self.num_batches = self.train.shape[0]
        self.num_test_batches = self.test.shape[0]
        self.num_val_batches = self.val.shape[0]
        print('Variable Means:', np.mean(self.train, axis=tuple(range(len(self.train.shape) - 1))))
        print('Variable STDs:', np.std(self.train, axis=tuple(range(len(self.train.shape) - 1))))
        self.subset = 'train'
        
    def sample_grid(self, flow_features, index, num_batches, timesteps=1, hist=1, start_index=0):
        b = index % num_batches
        t = index // num_batches + start_index
        features = flow_features[b][..., t:t+hist, :]
        targets = flow_features[b][..., t+hist:t+hist+timesteps, :]

        #shape into feature matrics
        features = torch.from_numpy(features.astype(np.float32)).requires_grad_()
        targets = torch.from_numpy(targets.astype(np.float32)).requires_grad_()

        if self.use_cuda:
            features = features.to(device='cuda')
            targets = targets.to(device='cuda')

        return features, targets

    def training(self):
        self.subset = 'train'

    def testing(self, test=False):
        if test:
            self.subset = 'test'
        else:
            self.subset = 'val'

    def long(self):
        self.subset = 'long'

    def dilate_dt(self, dilation):
        self.spacing[-1] /= dilation

    def set_start_index(self, start_index):
        self.start_index = start_index

    def __getitem__(self, index):
        if self.subset == 'train':
            return self.sample_grid(self.train, index, self.num_batches, timesteps=self.timesteps, hist=self.history, start_index=self.start_index)
        elif self.subset == 'val':
            return self.sample_grid(self.val, index, self.num_val_batches, timesteps=self.timesteps, hist=self.history, start_index=self.start_index)
        elif self.subset == 'long':
            return self.sample_grid(self.solution, index, len(self.solution), timesteps=self.duration - self.history, hist=self.history, start_index=self.start_index)
        elif self.subset == 'test':
            return self.sample_grid(self.test, index, self.num_test_batches, timesteps=self.timesteps, hist=self.history, start_index=self.start_index)

    def __len__(self):
        if self.subset == 'train':
            return (self.solution.shape[-2] - self.timesteps - self.history - self.start_index + 1) * self.num_batches
        elif self.subset == 'val':
            return (self.solution.shape[-2] - self.timesteps - self.history - self.start_index + 1) * self.num_val_batches
        elif self.subset == 'long':
            return len(self.solution)
        elif self.subset == 'test':
            return (self.solution.shape[-2] - self.timesteps - self.history - self.start_index + 1) * self.num_test_batches


class PDEBenchGrid(DynamicGrid):
    def __init__(self, *args, timesteps=1, history=0, use_cuda=False, **kwargs):
        grid, solution = self.read_input_directory(*args, **kwargs)
        super(PDEBenchGrid, self).__init__(grid, solution, timesteps=timesteps, history=history, use_cuda=use_cuda)
    
    def read_input_directory(self, filename,
                                   saved_folder='',
                                   reduced_resolution=1,
                                   reduced_resolution_t=1,
                                   reduced_batch=1,
                                   conservative_form=False
                                   ):
        """
        :param filename: filename that contains the dataset
        :type filename: STR
        """
        
        # Define path to files
        root_path = os.path.abspath(saved_folder + filename)
        assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'
        
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            keys.sort()
            print(keys)
            vx = f['Vx']
            vy = f['Vy']
            # vz = f['Vz']
            density = f['density']
            pressure = f['pressure']
            t = f['t-coordinate']
            x = f['x-coordinate']
            y = f['y-coordinate']
            # z = f['z-coordinate']

            vx = np.array(vx, dtype=np.float32)
            vy = np.array(vy, dtype=np.float32)
            # vz = np.array(vz, dtype=np.float32)
            density = np.array(density, dtype=np.float32)
            pressure = np.array(pressure, dtype=np.float32)

            t = np.array(t, dtype=np.float32)    ###, t, x are equispaced
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            # z = np.array(z, dtype=np.float32)

            X, Y = np.meshgrid(x, y)
            self.grid = np.stack((X, Y), axis=-1)
            self.data = np.stack([density, pressure, vx, vy],axis=-1).transpose(0,2,3,1,4)

            if reduced_batch > 1:
                self.data = self.data[::reduced_batch,...]
            
            # reduce to 128 for all experiments
            if reduced_resolution > 1:
                print('Setting resolution reduction is deprecated, using fixed reduction to 128.')
            
            reduced_resolution = self.data.shape[1] // 128
            if reduced_resolution > 1:
                print('Reducing spatial resolution by factor of ', reduced_resolution)
                self.grid = self.grid[::reduced_resolution, ::reduced_resolution]
                # self.data = self.data[:, ::reduced_resolution, ::reduced_resolution, ...]
                new_shape = (self.data.shape[1] // reduced_resolution,
                                self.data.shape[2] // reduced_resolution)
                B, H, W, T, C = self.data.shape
                x = self.data.reshape((B, H, W, -1)).transpose(0, 3, 1, 2)
                x = F.interpolate(torch.from_numpy(x), size=new_shape, mode='bilinear').permute(0, 2, 3, 1)
                self.data = x.reshape((B, *new_shape, T, C)).numpy()
                print(self.data.shape)   # B, X, Y, T, C
        
        if conservative_form:
            print('Converting to conservative form...')
            self.data = convert_to_conservative_form(self.data)

        return self.grid, self.data
    
class PINOGrid(DynamicGrid):
    def __init__(self, *args, timesteps=1, history=0, use_cuda=False, **kwargs):
        grid, solution  = self.read_input_directory(*args, **kwargs)
        super(PINOGrid, self).__init__(grid, solution, timesteps=timesteps, history=history, use_cuda=use_cuda)
    
    def read_input_directory(self, filename):
        mat = {}
        f = h5py.File(filename)
        for k, v in f.items():
            mat[k] = np.array(v)
        w = mat['u'].transpose((3, 1, 2, 0))[...,None]
        grid = mat['a'][..., 0, None]
        grid = np.repeat(grid, 2, axis=-1)
        w = w / w.std()
        return grid, w
    
