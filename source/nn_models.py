import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from .layers.fourier import SpectralConv2d_fast
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
        #x_out_total = torch.sum(x[:, 0])
        #x_scaled = x[:, 0] * (x_in_total / x_out_total)
        #x = torch.cat([x_scaled[..., None], x[:, 1:]], dim=-1)

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
