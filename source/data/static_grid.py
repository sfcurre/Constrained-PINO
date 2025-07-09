import numpy as np
import scipy.sparse as sp
import glob, os, h5py
import math as mt
import torch
from torch.utils.data import Dataset

from .dynamic_grid import *

class StaticGrid(DynamicGrid):
    def __init__(self, grid, solution, timesteps=1, history=1, use_cuda=False):
        super(StaticGrid, self).__init__(grid, solution, timesteps=timesteps, history=history, use_cuda=use_cuda)
    
    def sample_grid(self, flow_features, index, num_batches, timesteps=1, hist=1):
        locations, features, targets = super().sample_grid(flow_features, index, num_batches, timesteps=timesteps, hist=hist)
        features = features.reshape(features.shape[:-2] + (-1,))

        features = features.tile((timesteps, 1))
        locations = locations.unsqueeze(-2)
        locations = locations.tile((timesteps, 1))
        return locations, features, targets    

class PDEBenchStaticGrid(StaticGrid, PDEBenchGrid):
    def __init__(self, *args, timesteps=1, history=1, use_cuda=False, **kwargs):
        grid, solution = self.read_input_directory(*args, **kwargs)
        super(PDEBenchStaticGrid, self).__init__(grid, solution, timesteps=timesteps, history=history, use_cuda=use_cuda)
        
class PINOStaticGrid(StaticGrid, PINOGrid):
    def __init__(self, *args, timesteps=1, history=1, use_cuda=False, **kwargs):
        grid, solution  = self.read_input_directory(*args, **kwargs)
        super(PINOGrid, self).__init__(grid, solution, timesteps=timesteps, history=history, use_cuda=use_cuda)
