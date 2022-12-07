import numpy as np
import scipy.sparse as sp
import glob
import torch
from torch_geometric.utils import get_laplacian, coalesce
from torch.utils.data import Dataset

# MINIBATCH EVERYTHING with collate_fn

class DynamicGraph(Dataset):
    def __init__(self, grid, solution, timesteps=1, use_cuda=False):
        self.grid = grid
        self.solution = solution
        self.timesteps = timesteps
        self.use_cuda = use_cuda
        self.nfeats = self.solution.shape[-1]
        self.ndims = self.grid.shape[-1]
        
        self.locations = self.grid.reshape((-1, self.grid.shape[-1]))
        self.adj = self.build_adj_matrix()
        
        self.train = self.solution[:int(.8 * len(self.solution))]
        self.test = self.solution[int(.8 * len(self.solution)):]
        self.is_training = True

    def build_adj_matrix(self):
        #edgelist should be the same for both source and target
        edges = self.construct_edgelist(*self.grid.shape[:3], periodic=False)
        return get_laplacian(torch.tensor(edges), torch.ones(edges.shape[1]),
                             normalization='sym')
        
    def sample_graph(self, flow_features, t, tdif=1):
        features, targets = flow_features[t], flow_features[t+1: t+tdif+1]
        
        #shape into feature matrics
        features = features.reshape((-1, features.shape[-1]))
        targets = targets.reshape((targets.shape[0], -1, targets.shape[-1]))

        features = torch.from_numpy(features.astype(np.float32)).requires_grad_()
        targets = torch.from_numpy(targets.astype(np.float32)).requires_grad_()
        locations = torch.from_numpy(self.locations.astype(np.float32)).requires_grad_()
        mask = None

        adj = self.adj
        if self.use_cuda:
            adj = (adj[0].cuda(), adj[1].cuda())
            locations = locations.cuda()
            features = features.cuda()
            targets = targets.cuda()

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
            return len(self.train) - self.timesteps
        else:
            return len(self.test) - self.timesteps

    @staticmethod
    def construct_edgelist(xdim, ydim, zdim, periodic=False):
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
