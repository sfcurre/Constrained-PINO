import numpy as np
import scipy.sparse as sp
import glob, os, h5py
import math as mt
import torch
from torch.utils.data import Dataset

class DynamicGrid(Dataset):
    def __init__(self, grid, solution, timesteps=1, history=1, use_cuda=False):
        self.grid = grid
        # batch, x, y, z, t, ch
        self.solution = solution
        self.duration = self.solution.shape[-2]
        self.spacing = [np.diff(self.grid[...,i], axis=i).mean() for i in range(len(self.grid.shape) - 1)]
        self.spacing.append(1 / (self.duration-1))
        
        self.timesteps = min(timesteps, self.duration - history)
        self.history = history
        self.use_cuda = use_cuda
        self.num_channels = self.solution.shape[-1]
        self.nfeats = self.num_channels
        self.ndims = self.grid.shape[-1]

        self.train = self.solution[:int(.9 * self.solution.shape[0])]
        self.test = self.solution[int(.9 * self.solution.shape[0]):]
        self.num_batches = self.train.shape[0]
        self.num_test_batches = self.test.shape[0]
        self.variable_weights = 1
        self.is_training = True
        
    def sample_grid(self, flow_features, index, num_batches, timesteps=1, hist=1):
        b = index % num_batches
        t = index // num_batches
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
        self.is_training = True

    def testing(self):
        self.is_training = False

    def __getitem__(self, index):
        if self.is_training:
            return self.sample_grid(self.train, index, self.num_batches, timesteps=self.timesteps, hist=self.history)
        else:
            return self.sample_grid(self.test, index, self.num_test_batches, timesteps=self.timesteps, hist=self.history)

    def __len__(self):
        if self.is_training:
            return (self.solution.shape[-2] - self.timesteps - self.history + 1) * self.num_batches
        else:
            return (self.solution.shape[-2] - self.timesteps - self.history + 1) * self.num_test_batches


class PDEBenchGrid(DynamicGrid):
    def __init__(self, *args, timesteps=1, history=0, use_cuda=False, **kwargs):
        grid, solution = self.read_input_directory(*args, **kwargs)
        super(PDEBenchGrid, self).__init__(grid, solution, timesteps=timesteps, history=history, use_cuda=use_cuda)
    
    def read_input_directory(self, filename,
                                   saved_folder='',
                                   reduced_resolution=1,
                                   reduced_resolution_t=1,
                                   reduced_batch=1,
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
            if 'tensor' not in keys:
                _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                idx_cfd = _data.shape
                if len(idx_cfd)==3:  # 1D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                          3],
                                        dtype=np.float32)
                    #density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,2] = _data   # batch, x, t, ch

                    self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                    self.grid = self.grid[::reduced_resolution, None]

                if len(idx_cfd)==4:  # 2D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          idx_cfd[3]//reduced_resolution,
                                          mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                          4],
                                         dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,2] = _data   # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,3] = _data   # batch, x, t, ch

                    x = np.array(f["x-coordinate"], dtype=np.float32)
                    y = np.array(f["y-coordinate"], dtype=np.float32)
                    X, Y = np.meshgrid(x, y)
                    self.grid = np.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]
            
                if len(idx_cfd)==5:  # 3D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          idx_cfd[3]//reduced_resolution,
                                          idx_cfd[4]//reduced_resolution,
                                          mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                          5],
                                         dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,2] = _data   # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,3] = _data   # batch, x, t, ch
                    # Vz
                    _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,4] = _data   # batch, x, t, ch

                    x = np.array(f["x-coordinate"], dtype=np.float32)
                    y = np.array(f["y-coordinate"], dtype=np.float32)
                    z = np.array(f["z-coordinate"], dtype=np.float32)
                    X, Y, Z = np.meshgrid(x, y, z)
                    self.grid = np.stack((X, Y, Z), axis=-1)[::reduced_resolution,\
                                                             ::reduced_resolution,\
                                                             ::reduced_resolution]
                           
            else:  # scalar equations
                ## data dim = [t, x1, ..., xd, v]
                _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                if len(_data.shape) == 3:  # 1D
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data = _data[:, :, :, None]  # batch, x, t, ch

                    self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                    self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                if len(_data.shape) == 4:  # 2D Darcy flow
                    # u: label
                    _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    #if _data.shape[-1]==1:  # if nt==1
                    #    _data = np.tile(_data, (1, 1, 1, 2))
                    self.data = _data
                    # nu: input
                    _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    self.data = np.concatenate([_data, self.data], axis=-1)
                    self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch

                    x = np.array(f["x-coordinate"], dtype=np.float32)
                    y = np.array(f["y-coordinate"], dtype=np.float32)
                    X, Y = np.meshgrid(x, y)
                    self.grid = np.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

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
    
