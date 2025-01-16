import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveTransform(nn.Module):
    """
    Simple wave_transform
    """
    def __init__(self, cos=True, sin=True):
        self.cos = cos
        self.sin = sin
        super(WaveTransform, self).__init__()
        
    def forward(self, to_transform, concat_to=None):
        tensors = []
        if concat_to is not None:
            tensors.append(concat_to)
        if self.cos:
            tensors.append(torch.cos(to_transform))
        if self.sin:
            tensors.append(torch.sin(to_transform))
        return torch.cat(tensors, dim=-1)
