import numpy as np
import scipy.sparse as sp
import glob
import torch
from torch_geometric.utils import get_laplacian, coalesce
from torch.utils.data import Dataset

# MINIBATCH EVERYTHING with collate_fn

class DynamicGraph(Dataset):
    def __init__(self, grid, solution, num_channels, timesteps=1, use_cuda=False):
        self.grid = grid
        self.solution = solution
        self.num_channels = num_channels
        self.timesteps = timesteps
        self.use_cuda = use_cuda
        self.nfeats = self.num_channels
        self.ndims = self.grid.shape[-1]
        
        self.locations = self.grid.reshape((-1, self.grid.shape[-1]))
        self.adj = self.build_adj_matrix()
        
        self.train = self.solution[..., :int(.9 * self.solution.shape[-1])]
        self.test = self.solution[..., int(.9 * self.solution.shape[-1]):]
        self.num_batches = self.train[0].shape[-1] // self.num_channels
        self.is_training = True

    def build_adj_matrix(self):
        #edgelist should be the same for both source and target
        edges = self.construct_edgelist(*self.grid.shape[:3], periodic=False)
        return get_laplacian(torch.tensor(edges), torch.ones(edges.shape[1]),
                             normalization='sym')
        
    def sample_graph(self, flow_features, index, tdif=1):
        t = index // self.num_batches
        c = (index % self.num_batches) * self.num_channels
        features = flow_features[t][..., c: c + self.num_channels]
        targets = flow_features[t+1: t+tdif+1][..., c: c + self.num_channels]

        #shape into feature matrics
        features = features.reshape((-1, features.shape[-1]))
        targets = targets.reshape((targets.shape[0], -1, targets.shape[-1]))

        features = torch.from_numpy(features.astype(np.float32)).requires_grad_()
        targets = torch.from_numpy(targets.astype(np.float32)).requires_grad_()
        locations = torch.from_numpy(self.locations.astype(np.float32)).requires_grad_()
        mask = None

        adj = self.adj
        if self.use_cuda:
            adj = (adj[0].to(device='cuda'), adj[1].to(device='cuda'))
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
            return self.sample_graph(self.train, index, tdif=self.timesteps)
        else:
            return self.sample_graph(self.test, index, tdif=self.timesteps)

    def __len__(self):
        if self.is_training:
            return (len(self.train) - self.timesteps) * self.num_batches
        else:
            return (len(self.test) - self.timesteps) * self.num_batches

    @staticmethod
    def construct_edgelist(xdim, ydim, zdim, periodic=False):
        print(xdim, ydim, zdim)
        mode = 'wrap' if periodic else 'clip'
        getnode = lambda x, y, z: np.ravel_multi_index((x, y, z), (xdim, ydim, zdim), mode=mode)
        edges = []
        for x in range(xdim):
            for y in range(ydim):
                for z in range(zdim):
                    node = getnode(x, y, z)
                    edges.append([node, node])
                    
                    node_x_1 = getnode(x - 1, y, z)
                    edges.append([node_x_1, node])
                    edges.append([node, node_x_1])
                    
                    node_y_1 = getnode(x, y - 1, z)
                    edges.append([node_y_1, node])
                    edges.append([node, node_y_1])
                    
                    node_z_1 = getnode(x, y, z - 1)
                    edges.append([node_z_1, node])
                    edges.append([node, node_z_1])

        return np.array(edges, dtype=np.int64).T
