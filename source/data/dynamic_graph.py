import numpy as np
import scipy.sparse as sp
import glob
import torch
from torch_geometric.utils import get_laplacian
from torch.utils.data import Dataset

class DynamicGraph(Dataset):
    def __init__(self, grid, solution, timesteps=0, use_cuda=False):
        self.grid = grid
        self.solution = solution
        self.timesteps = timesteps
        self.use_cuda = use_cuda
        
        self.locations = self.grid.reshape((-1, self.grid.shape[-1]))
        self.adj = self.build_adj_matrix()
        
        self.train = self.solution[:int(.8 * len(self.solution))]
        self.test = self.solution[int(.8 * len(self.solution)):]
        self.training = True

    def build_adj_matrix(self):
        #edgelist should be the same for both source and target
        edges = self.construct_edgelist(*self.grid.shape[:3], periodic=False)
        return get_laplacian(torch.tensor(edges), torch.ones(edges.shape[1]),
                             normalization='sym')
        
    def sample_graph(self, flow_features, t, tdif=0):
        features, targets = flow_features[t], flow_features[t+1: t+tdif+1]
        
        #shape into feature matrics
        features = features.reshape((-1, features.shape[-1]))
        targets = targets.reshape((targets.shape[0], -1, targets.shape[-1]))

        features = torch.from_numpy(features.astype(np.float32)).requires_grad_()
        targets = torch.from_numpy(targets.astype(np.float32)).requires_grad_()
        locations = torch.from_numpy(self.locations.astype(np.float32)).requires_grad_()

        adj = self.adj
        if self.use_cuda:
            adj = (adj[0].cuda(), adj[1].cuda())
            locations = locations.cuda()
            features = features.cuda()
            targets = targets.cuda()

        return adj, locations, features, targets

    def training(self):
        self.training = True

    def testing(self):
        self.training = False

    def __getitem__(self, index):
        if self.training:
            return self.sample_graph(self.train, index, tdif=self.timesteps)
        else:
            return self.sample_graph(self.test, index, tdif=self.timesteps)

    def __len__(self):
        if self.training:
            return len(self.train) - self.timesteps - 1
        else:
            return len(self.test) - self.timesteps - 1

    @staticmethod
    def construct_edgelist(xdim, ydim, zdim, periodic=False):
        getnode = lambda x, y, z: x * ydim * zdim + y * zdim + z
        edges = []
        for x in range(xdim):
            for y in range(ydim):
                for z in range(zdim):
                    node = getnode(x, y, z)
                    edges.append([node, node])
                    if x > 0:
                        node_x_1 = getnode(x - 1, y, z)
                        if node_x_1 < node:
                            edges.append([node_x_1, node])
                            edges.append([node, node_x_1]) 
                    if y > 0:
                        node_y_1 = getnode(x, y - 1, z)
                        if node_y_1 < node:
                            edges.append([node_y_1, node])
                            edges.append([node, node_y_1])
                    if z > 0:
                        node_z_1 = getnode(x, y, z - 1)
                        if node_z_1 < node:
                            edges.append([node_z_1, node])
                            edges.append([node, node_z_1])
        if periodic:
            # add connections from (ydim - 1) to 0
            for x in range(xdim):
                for z in range(zdim):
                    n0 = getnode(x, 0, z)
                    n1 = getnode(x, ydim - 1, z)
                    edges.append([n0, n1])
                    edges.append([n1, n0])
            # add connections from (xdim - 1) to 0
            for y in range(ydim):
                for z in range(zdim):
                    n0 = getnode(0, y, z)
                    n1 = getnode(xdim - 1, y, z)
                    edges.append([n0, n1])
                    edges.append([n1, n0])
            # add connections from (zdim - 1) to 0
            for x in range(xdim):
                for y in range(ydim):
                    n0 = getnode(x, y, 0)
                    n1 = getnode(x, y, zdim - 1)
                    edges.append([n0, n1])
                    edges.append([n1, n0])

        return np.array(edges, dtype=np.int64).T
