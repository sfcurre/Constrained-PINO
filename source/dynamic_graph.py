import numpy as np
import scipy.sparse as sp
import glob
import torch
from torch_geometric.utils import get_laplacian

class DynamicGraph3D:
    def __init__(self, grd_path, solution_path, periodic=False, use_cuda=False):
        if not grd_path.endswith('.npy') and not solution_path.endswith('.npy'):
            self.grid, self.solution = self.read_input_directory(grd_path, solution_path)
            np.save(grd_path.replace('.', '_') + '.npy', self.grid)
            np.save(solution_path + '_sol.npy', self.solution)
        else:
            self.grid, self.solution = np.load(grd_path), np.load(solution_path)
        self.periodic = periodic
        if self.periodic:
            self.grid, self.solution = self.cut_periodic(self.grid, self.solution)
            trig_grid = self.periodicize_grid(self.grid)
            self.locations = trig_grid.reshape((-1, trig_grid.shape[-1]))    
        else:
            self.locations = self.grid.reshape((-1, self.grid.shape[-1]))
        self.adj = self.build_adj_matrix()
        self.train = self.solution[:int(.8 * len(self.solution))]
        self.test = self.solution[int(.8 * len(self.solution)):]

    def read_input_directory(self, grd_path, solution_path):
        locations = self.read_grd_data(grd_path)

        flow_features = []
        for filename in sorted(glob.glob(solution_path + '*')):
            if not filename.endswith('.grd'):
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

    def periodicize_grid(self, grid):
        #new coordinate system
        #scale invariant, will produce wrap-around coordinates when combined with WaveTranform
        xcoords = np.linspace(0, 2*np.pi, grid.shape[0])
        ycoords = np.linspace(0, 2*np.pi, grid.shape[1])
        zcoords = np.linspace(0, 2*np.pi, grid.shape[2])

        x, y, z = np.meshgrid(xcoords, ycoords, zcoords)
        trig_grid = np.stack([x, y, z], axis = -1)
        return trig_grid

    def build_adj_matrix(self):
        #edgelist should be the same for both source and target
        edges = self.construct_edgelist(*self.grid.shape[:3], periodic=self.periodic)
        # edges, weights = get_laplacian(torch.tensor(edges), torch.ones(edges.shape[1]), normalization='sym')
        # adj = torch.sparse_coo_tensor(edges, weights)
        # return adj
        return get_laplacian(torch.tensor(edges), torch.ones(edges.shape[1]),
                             normalization='sym')
        
    def sample_graph(self, flow_features, t=None, tdif=1):
        if t is None:
            t = np.random.randint(len(flow_features) - tdif)
        features, targets = flow_features[t], flow_features[t+1: t+tdif+1]
        
        #shape into feature matrics
        features = features.reshape((-1, features.shape[-1]))
        targets = targets.reshape((targets.shape[0], -1, targets.shape[-1]))

        features = torch.from_numpy(features.astype(np.float32)).requires_grad_()
        targets = torch.from_numpy(targets.astype(np.float32)).requires_grad_()
        locations = torch.from_numpy(self.locations.astype(np.float32)).requires_grad_()

        if self.use_cuda:
            adj = (self.adj[0].cuda(), self.adj[1].cuda())
            locations = locations.cuda()
            features = features.cuda()
            targets = targets.cuda()

        return adj, locations, features, targets

    def sample_train(self, t=None, generator=False, timesteps=1):
        if not generator:
            return self.sample_graph(self.train, t=t, tdif=timesteps)
        ts = np.random.permutation(len(self.train))
        for t in ts:
            yield self.sample_graph(self.train, t=t, tdif=timesteps)

    def sample_test(self, t=None, generator=False, timesteps=1):
        if not generator:
            return self.sample_graph(self.test, t=t, tdif=timesteps)
        for t in range(len(self.test)):
            yield self.sample_graph(self.test, t=t, tdif=timesteps)

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
