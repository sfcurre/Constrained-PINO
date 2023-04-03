import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from .layers.fourier import FNO2d
from .layers.wave_transform import WaveTransform
# from pytorch_memlab import profile


class PIGNN(nn.Module):
    def __init__(self, nfeat, ndims, nhid, nlayers, nclass, dropout, periodic=True):
        super(PIGNN, self).__init__()
        self.dropout = dropout
        self.periodic = True

        print(ndims, nfeat, nhid)

        if self.periodic:
            self.wave_transform = WaveTransform()
            ndims *= 2
        
        self.linear_in = tgnn.Linear(nfeat + ndims, nhid)

        self.gcns = []
        for i in range(nlayers):
            gcn = tgnn.GraphConv(nhid, nhid)
            self.gcns.append(gcn)

        self.linear_out = tgnn.Linear(nhid, nclass)

    # @profile
    def forward(self, locations, features, adj):
        x_in_total = torch.sum(features[:, 0])
        if self.periodic:
            x = self.wave_transform(locations, concat_to=features)
        else:
            x = torch.cat([locations, features], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.linear_in(x)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        for gcn in self.gcns:
            x = gcn(x, *adj)
            x = torch.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.linear_out(x)
        return x

class PIFNO2d(nn.Module):
    def __init__(self, ndims, nclass, periodic=True, history=10):
        super(PIFNO2d, self).__init__()
        self.ndims = ndims
        self.nclass = nclass
        self.periodic = True
        self.history = history

        if self.periodic:
            self.wave_transform = WaveTransform()
            self.ndims *= 2
        
        self.fno = FNO2d(self.nclass, self.ndims, initial_step=self.history)

    # @profile
    def forward(self, locations, features, adj):
        total_density = torch.sum(features[..., -4])
        total_pressure = torch.sum(features[..., -1])

        if self.periodic:
            locations = self.wave_transform(locations, concat_to=None)

        features = features[None, ..., 0, :]
        locations = locations[None, ..., 0, :]
        x = self.fno(features, locations)[0]

        out_density = torch.sum(x[..., 0])
        scaled_density = x[..., :1] * total_density / out_density

        out_pressure = torch.sum(x[..., 3])        
        scaled_pressure = x[..., 3:] * total_pressure / out_pressure
        
        x = torch.concat([scaled_density, x[..., 1:3], scaled_pressure], dim=-1) 
        #x = torch.concat([scaled_density, x[..., 1:]], dim=-1) 

        return x

class SpGAT(nn.Module):
    def __init__(self, nfeat, ndims, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.att = tgnn.GATConv(nfeat + ndims,
                                nhid,
                                nheads,
                                dropout=dropout,
                                negative_slope=alpha,
                                concat=True)
        self.out_att = tgnn.GATConv(nhid * nheads,
                                    nclass,
                                    dropout=dropout,
                                    negative_slope=alpha,
                                    concat=False)
        
    def forward(self, locations, features, adj):
        x = torch.cat([locations, features], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.att(x, *adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, *adj)
        return x

class PeriodicSpGAT(nn.Module):
    def __init__(self, nfeat, ndims, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(PeriodicSpGAT, self).__init__()
        self.dropout = dropout

        self.wave_transform = WaveTransform()
        self.att = tgnn.GATConv(nfeat + 2*ndims,
                                nhid,
                                nheads,
                                dropout=dropout,
                                negative_slope=alpha,
                                concat=True)
        self.out_att = tgnn.GATConv(nhid * nheads,
                                    nclass,
                                    dropout=dropout,
                                    negative_slope=alpha,
                                    concat=False)
        
    def forward(self, locations, features, adj):
        x = self.wave_transform(locations, concat_to=features)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.att(x, *adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, *adj)
        return x

class DeepSpGAT(nn.Module):
    def __init__(self, nfeat, ndims, nhid, natt, nclass, dropout, alpha, nheads, periodic=True):
        """Sparse version of GAT."""
        super(DeepSpGAT, self).__init__()
        self.dropout = dropout
        self.periodic = True

        if self.periodic:
            self.wave_transform = WaveTransform()
            ndims *= 2
        

        self.att = tgnn.GATConv(nfeat + ndims,
                                nhid,
                                nheads,
                                dropout=dropout,
                                negative_slope=alpha,
                                concat=False)
        
        self.internal_atts = []
        for i in range(natt - 2):
            self.internal_atts.append(tgnn.GATConv(nhid,
                                      nhid,
                                      nheads,
                                      dropout=dropout,
                                      negative_slope=alpha,
                                      concat=False))

        self.out_att = tgnn.GATConv(nhid,
                                    nclass,
                                    dropout=dropout,
                                    negative_slope=alpha,
                                    concat=False)
        
    # @profile
    def forward(self, locations, features, adj):
        if self.periodic:
            x = self.wave_transform(locations, concat_to=features)
        else:
            x = torch.cat([locations, features], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.att(x, *adj)
        for att in self.internal_atts:
            x = F.dropout(x, self.dropout, training=self.training)
            x = att(x, *adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, *adj)
        return x
