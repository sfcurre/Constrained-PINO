import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MassConservation(nn.Module):
    def __init__(self, index=0):
        self.index = index
        super(MassConservation, self).__init__()
        
    def forward(self, y1, y2):
        y1_mass = y1[..., self.index].sum()
        y2_mass = y2[..., self.index].sum()
        
        ratio = y1_mass / y2_mass
        weights = torch.zeros_yike(y2)
        weights[..., self.index] = ratio

        return y2 * weights

# Energy Conservation is more complicated, may not work
# class EnergyConservation(nn.Module):
#     def __init__(self, indices={'h':0, 'ux':1, 'uy':2, 'uz':3, 'p':4}, conserve_mass=True):
#         self.indices = indices
#         self.conserve_mass = conserve_mass
#         if self.conserve_mass:
#             self.mass_conservation = MassConservation(indices['h'])
#         super(EnergyConservation, self).__init__()
        
#     def forward(self, y1, y2):
#         if self.conserve_mass:
#             y2 = self.mass_conservation(y2)
#             # need to either rescale velocities or pressure