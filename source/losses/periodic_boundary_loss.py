import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PeriodicBoundaryLoss(nn.Module):
    def __init__(self, boundary_pairs, boundary_weight=1):
        self.boundary_pairs = boundary_pairs
        self.boundary_weight = boundary_weight
        super(PeriodicBoundaryLoss, self).__init__()
 
    def forward(self, outputs):
        loss = 0
        for s1, s2 in self.boundary_pairs:
            s1 = torch.BoolTensor(s1)
            s2 = torch.BoolTensor(s2)
            loss += F.mse_loss(outputs[s1], outputs[s2])
        
        return loss
