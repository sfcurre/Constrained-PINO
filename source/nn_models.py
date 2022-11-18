import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from .layers.fourier import SpectralConv2d_fast
from .layers.wave_transform import WaveTransform

# Code collected from https://github.com/Diego999/pyGAT

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
