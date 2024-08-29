import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from .layers.fourier import FNO1d, FNO2d, FNO3d
from .layers.wave_transform import WaveTransform
# from pytorch_memlab import profile

class PIFNO(nn.Module):
    def __init__(self, ndims, nclass, periodic=True, scale_mass=True, scale_momentum=False, scale_energy=False, history=10):
        super(PIFNO, self).__init__()
        self.ndims = ndims
        self.nclass = nclass
        self.periodic = periodic
        self.scale_mass = scale_mass
        self.scale_momentum = scale_momentum
        self.scale_energy = scale_energy
        self.history = history

        if self.periodic:
            self.wave_transform = WaveTransform()
            self.ndims *= 2
        
        if ndims == 1:
            self.fno = FNO1d(self.nclass, self.ndims, initial_step=self.history)
        if ndims == 2:
            self.fno = FNO2d(self.nclass, self.ndims, initial_step=self.history)
        if ndims == 3:
            self.fno = FNO3d(self.nclass, self.ndims, initial_step=self.history)

    def scale_totals(self, features, x):
        total_density = torch.sum(features[..., -self.nclass])
        total_pressure = torch.sum(features[..., -1])
        
        out_density = torch.sum(x[..., 0])
        scaled_density = x[..., :1] * total_density / out_density

        out_pressure = torch.sum(x[..., -1])
        scaled_pressure = x[..., -1:] * total_pressure / out_pressure
        
        if self.scale_momentum:
            all_but_last_dim = list(range(len(features.shape) - 1))
            total_momentums = torch.sum(features[..., -self.nclass:-self.nclass+1]
                                        * features[..., -self.nclass + 1:-1], dim=all_but_last_dim)
            out_momentums = torch.sum(scaled_density * x[..., 1:-1], dim=all_but_last_dim)
            momentum_ratio = total_momentums / out_momentums
            scaled_velocity = x[..., 1:-1] * momentum_ratio
            x = torch.concat([scaled_density, scaled_velocity, scaled_pressure], dim=-1)

        elif self.scale_energy:
            total_kinetic_energy = torch.sum(features[..., -self.nclass: -self.nclass + 1]
                                                  * features[..., -self.nclass + 1:-1] ** 2)
            out_kinetic_energy = torch.sum(scaled_density * x[..., 1:-1] ** 2)
            ke_ratio = total_kinetic_energy / out_kinetic_energy
            # v2 = (vx^2 + vy^2 + vz^2) needs to be scaled by ke
            # C*v2 = Cvx^2 + Cvy^2 + Cvz^2
            # C*v2 = (root(C)vx)^2 + ...
            scaled_velocity = x[..., 1:-1] * torch.sqrt(ke_ratio)
            x = torch.concat([scaled_density, scaled_velocity, scaled_pressure], dim=-1)
                
            #print(torch.sum(x[...,0]), torch.sum(x[..., :1] * x[..., 1:-1]**2), torch.sum(x[..., -1]))
        else:
            x = torch.concat([scaled_density, x[..., 1:-1], scaled_pressure], dim=-1)

        return x

    def forward(self, locations, features):
        if self.periodic:
            locations = self.wave_transform(locations, concat_to=None)

        features = features.squeeze()[None, ...]
        locations = locations.squeeze()[None, ...]
        x = self.fno(features, locations)[0]

        if self.scale_mass:
            x = self.scale_totals(features, x)

        return x

class PIFNO_grad(PIFNO):
    def __init__(self, *args, **kwargs):
        super(PIFNO_grad, self).__init__(*args, **kwargs)
        
    def scale_grads(self, features, dx):
        
        scaled_density = dx[..., :1] - dx[..., :1].mean()
        scaled_pressure = dx[..., -1:] - dx[..., -1:].mean()
        
        if self.scale_momentum:
            all_but_last_dim = list(range(len(features.shape) - 1))
            # sum of -grad(rho)*u
            numerator = torch.sum(-scaled_density * features[..., -self.nclass + 1:-1])
            denominator = torch.sum((features[..., -self.nclass:-self.nclass+1] + scaled_density) * dx[..., 1:-1])
            scaled_velocity = dx[..., 1:-1] * (numerator / denominator)
            dx = torch.concat([scaled_density, scaled_velocity, scaled_pressure], dim=-1)

        elif self.scale_energy:
            raise NotImplementedError('Energy scaling is not implemented yet for gradient models.')
            
        else:
            dx = torch.concat([scaled_density, dx[..., 1:-1], scaled_pressure], dim=-1)

        return dx
    
    def forward(self, locations, features):
        if self.periodic:
            locations = self.wave_transform(locations, concat_to=None)

        features = features.squeeze()[None, ...]
        locations = locations.squeeze()[None, ...]
        dx = self.fno(features, locations)[0]

        if self.scale_mass:
            dx = self.scale_grads(features, dx)

        x = features[..., -self.nclass:] + dx
        return x
