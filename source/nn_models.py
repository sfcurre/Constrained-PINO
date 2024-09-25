import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from .layers.fourier import FNO1d, FNO2d, FNO3d
from .layers.wave_transform import WaveTransform
# from pytorch_memlab import profile

class PIFNO(nn.Module):
    def __init__(self, ndims, nclass, periodic=True, scale_mass=True, scale_momentum=False, scale_energy=False, history=10, kernel_size=1, static=False):
        super(PIFNO, self).__init__()
        self.ndims = ndims
        self.nclass = nclass
        self.periodic = periodic
        self.scale_mass = scale_mass
        self.scale_momentum = scale_momentum
        self.scale_energy = scale_energy
        self.history = history
        self.kernel_size = kernel_size
        self.static = static

        if self.periodic:
            self.wave_transform = WaveTransform()
            self.ndims *= 2
        
        if static:
            if ndims == 1:
                self.fno = FNO2d(self.nclass, self.nclass, self.ndims, kernel_size=self.kernel_size)
            if ndims == 2:
                self.fno = FNO3d(self.nclass, self.nclass, self.ndims, kernel_size=self.kernel_size)
            if ndims == 3:
                raise NotImplementedError("Static models are not impelmented for 3D flows.")
        else:
            if ndims == 1:
                self.fno = FNO1d(self.nclass * self.history, self.nclass, self.ndims, kernel_size=self.kernel_size)
            if ndims == 2:
                self.fno = FNO2d(self.nclass * self.history, self.nclass, self.ndims, kernel_size=self.kernel_size)
            if ndims == 3:
                self.fno = FNO3d(self.nclass * self.history, self.nclass, self.ndims, kernel_size=self.kernel_size)
        
    def scale_totals(self, features, x):
        spatial_dims = list(range(0, len(features.shape) - 2)) #all but the last two dims
        initial = features[..., -1:, :] #get last timestep of input
        total_density = torch.sum(initial[..., 0], dim=spatial_dims)
        total_pressure = torch.sum(initial[..., -1], dim=spatial_dims)
        
        out_density = torch.sum(x[..., 0], dim=spatial_dims)
        scaled_density = x[..., :1] * total_density / out_density

        out_pressure = torch.sum(x[..., -1], dim=spatial_dims)
        scaled_pressure = x[..., -1:] * total_pressure / out_pressure
        
        if self.scale_momentum:
            total_momentums = torch.sum(initial[..., :1] * initial[..., 1:-1], dim=spatial_dims)
            out_momentums = torch.sum(scaled_density * x[..., 1:-1], dim=spatial_dims)
            momentum_ratio = total_momentums / out_momentums
            scaled_velocity = x[..., 1:-1] * momentum_ratio
            x = torch.concat([scaled_density, scaled_velocity, scaled_pressure], dim=-1)

        elif self.scale_energy:
            total_kinetic_energy = torch.sum(initial[..., :1] * initial[..., 1:-1] ** 2, dim=spatial_dims)
            out_kinetic_energy = torch.sum(scaled_density * x[..., 1:-1] ** 2, dim=spatial_dims)
            ke_ratio = total_kinetic_energy / out_kinetic_energy
            # v2 = (vx^2 + vy^2 + vz^2) needs to be scaled by ke
            # C*v2 = Cvx^2 + Cvy^2 + Cvz^2
            # C*v2 = (root(C)vx)^2 + ...
            scaled_velocity = x[..., 1:-1] * torch.sqrt(ke_ratio)
            x = torch.concat([scaled_density, scaled_velocity, scaled_pressure], dim=-1)
                
        else:
            x = torch.concat([scaled_density, x[..., 1:-1], scaled_pressure], dim=-1)

        return x

    def forward(self, locations, features):
        if self.periodic:
            locations = self.wave_transform(locations, concat_to=None)

        fno_features = features.squeeze()[None, ...]
        locations = locations.squeeze()[None, ...]
        if not self.static:
            fno_features = fno_features.reshape(fno_features.shape[:-2] + (-1,))
        x = self.fno(fno_features, locations)[0]

        if self.scale_mass:
            x = self.scale_totals(features, x)

        return x

class PIFNO_grad(PIFNO):
    def __init__(self, *args, **kwargs):
        super(PIFNO_grad, self).__init__(*args, **kwargs)
        
    def scale_grads(self, features, dx):
        spatial_dims = list(range(0, len(features.shape) - 2)) #all but the last two dims
        initial = features[..., -1:, :] #get last timestep of input
        
        scaled_density = dx[..., :1] - dx[..., :1].mean(dim=spatial_dims)
        scaled_pressure = dx[..., -1:] - dx[..., -1:].mean(dim=spatial_dims)
        
        if self.scale_momentum:
            # sum of -grad(rho)*u
            numerator = torch.sum(-scaled_density * initial[..., 1:-1], dim=spatial_dims)
            denominator = torch.sum((initial[..., :1] + scaled_density) * dx[..., 1:-1], dim=spatial_dims)
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

        fno_features = features.squeeze()[None, ...]
        locations = locations.squeeze()[None, ...]
        if not self.static:
            fno_features = fno_features.reshape(fno_features.shape[:-2] + (-1,))
        dx = self.fno(fno_features, locations)[0]

        if self.scale_mass:
            dx = self.scale_grads(features, dx)

        x = features[..., -self.nclass:] + dx
        return x


class iPIFNO(nn.Module):
    def __init__(self, *args, **kwargs):
        super(iPIFNO, self).__init__()
    
    def scale_totals(self, features, x):
        spatial_dims = list(range(0, len(features.shape) - 1)) #all but the last dim
        total_vorticity = torch.sum(features[..., -self.nclass])
        
        out_vorticity = torch.sum(x[..., 0])
        scaled_vorticity = x[..., :1] * total_vorticity / out_vorticity

        return scaled_vorticity

    def forward(self, locations, features):
        if self.periodic:
            locations = self.wave_transform(locations, concat_to=None)

        fno_features = features.squeeze()[None, ...]
        locations = locations.squeeze()[None, ...]
        if not self.static:
            fno_features = fno_features.reshape(fno_features.shape[:-2] + (-1,))
        x = self.fno(fno_features, locations)[0]

        if self.scale_mass:
            x = self.scale_totals(features, x)

        return x

