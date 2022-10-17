import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from .layers.attention import GraphAttentionLayer, SpGraphAttentionLayer
from .layers.fourier import SpectralConv2d_fast
from .layers.wave_transform import WaveTransform

# Code collected from https://github.com/Diego999/pyGAT

class PeriodicSpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(PeriodicSpGAT, self).__init__()
        self.dropout = dropout

        self.wave_transform = WaveTransform()
        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, locations, features, adj):
        x = self.wave_transform(locations, concat_to=features)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x


class PeriodicSpGATTorchGeo(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(PeriodicSpGATTorchGeo, self).__init__()
        self.dropout = dropout

        self.wave_transform = WaveTransform()
        self.att = tgnn.GATConv(nfeat + 6,
                                nhid,
                                nheads,
                                dropout=dropout,
                                negative_slope=alpha,
                                concat=True)
        self.att2 = tgnn.GATConv(nhid,
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
        x = self.att2(x, *adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, *adj)
        return x
