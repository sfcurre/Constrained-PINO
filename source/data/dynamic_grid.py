import numpy as np
import scipy.sparse as sp
import glob
import torch
from torch_geometric.utils import get_laplacian, coalesce
from torch.utils.data import Dataset

# MINIBATCH EVERYTHING with collate_fn

class DynamicGrid(Dataset):
    def __init__(self, grid, solution, num_channels, timesteps=1, history=0, use_cuda=False):
        self.grid = grid
        print(grid.shape)
        print(solution.shape)
        self.solution = solution
        self.num_channels = num_channels
        self.timesteps = timesteps
        self.history = history
        self.use_cuda = use_cuda
        self.nfeats = self.num_channels
        self.ndims = self.grid.shape[-1]
        
        self.train = self.solution[..., :int(.9 * self.solution.shape[-1] / self.num_channels) * self.num_channels]
        self.test = self.solution[..., int(.9 * self.solution.shape[-1] / self.num_channels) * self.num_channels:]
        self.num_batches = self.train.shape[-1] // self.num_channels
        self.num_test_batches = self.test.shape[-1] // self.num_channels
        self.is_training = True
    
    def sample_grid(self, flow_features, index, num_batches, tdif=1, hist=1):
        t = index // num_batches
        c = (index % num_batches) * self.num_channels
        features = flow_features[t: t+hist][..., c: c + self.num_channels]
        targets = flow_features[t+hist: t+hist+tdif][..., c: c + self.num_channels]

        #shape into feature matrics
        features = np.moveaxis(features, 0, -2)
        features = features.reshape(features.shape[:-2] + (-1,))

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
            return self.sample_grid(self.train, index, self.num_batches, tdif=self.timesteps, hist=self.history)
        else:
            return self.sample_grid(self.test, index, self.num_test_batches, tdif=self.timesteps, hist=self.history)

    def __len__(self):
        if self.is_training:
            return (len(self.train) - self.timesteps - self.history + 1) * self.num_batches
        else:
            return (len(self.test) - self.timesteps - self.history + 1) * self.num_test_batches


class PDEBenchGrid(DynamicGrid):
    def __init__(self, *args, timesteps=1, history=0, use_cuda=False, **kwargs):
        grid, solution, num_channels = self.read_input_directory(*args, **kwargs)
        super(PDEBenchGrid, self).__init__(grid, solution, num_channels,
                                           timesteps=timesteps, history=history, use_cuda=use_cuda)
    
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

        while len(self.grid.shape) < 4:
            self.grid = np.expand_dims(self.grid, axis=-2)
        while len(self.data.shape) < 6:
            self.data = np.expand_dims(self.data, axis=-3)

        self.data = self.data.transpose((-2, 1, 2, 3, 0, -1)) #t, x, y, batch, ch
        #self.data = (self.data - self.data.mean(axis=(0, 1, 2, 3, 4))) / self.data.std(axis=(0, 1, 2, 3, 4))
        #self.data = self.data / self.data.std(axis=(0, 1, 2, 3, 4))

        self.data[..., 0:1] = self.data[..., 0:1] / self.data[..., 0:1].std()
        self.data[..., 2:] = self.data[..., 2:] / self.data[..., 2:].std()
        self.data[..., 1:2] = self.data[..., 1:2] / self.data[..., 1:2].std()

        self.data = np.concatenate([self.data[..., 0:1], self.data[..., 2:], self.data[..., 1:2]], axis=-1)       

        num_channels = self.data.shape[-1]
        self.data = self.data.reshape(self.data.shape[:-2] + (-1,))

        return self.grid, self.data, num_channels

class TGVGrid(DynamicGrid):
    def __init__(self, grd_path, solution_path, timesteps=1, history=0, use_cuda=False):
        if not grd_path.endswith('.npy') and not solution_path.endswith('.npy'):
            self.grid, self.solution = TGVGraph.read_input_directory(self, grd_path, solution_path)
            np.save(grd_path.replace('.', '_') + '.npy', self.grid)
            np.save(solution_path + '_sol.npy', self.solution)
        else:
            self.grid, self.solution = np.load(grd_path), np.load(solution_path)
        self.grid, self.solution = self.cut_periodic(self.grid, self.solution)
        super(DynamicGrid, self).__init__(self.grid, self.solution, 5,
                                          timesteps=timesteps, history=history, use_cuda=use_cuda)

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