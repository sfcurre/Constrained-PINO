import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_memlab import profile

class PIFNO(nn.Module):
    def __init__(self, fno, scale_mass=True, scale_momentum=False, scale_energy=False, static=False, uncertainty=False, grad_features=False):
        super(PIFNO, self).__init__()
        self.fno = fno
        self.scale_mass = scale_mass
        self.scale_momentum = scale_momentum
        self.scale_energy = scale_energy
        self.static = static
        self.uncertainty = uncertainty
        self.grad_features = grad_features
            
    def scale_totals(self, initial, x):
        spatial_dims = list(range(1, len(initial.shape) - 2)) # all but the batch and last two dims
        
        total_density = torch.sum(initial[..., :1], dim=spatial_dims, keepdim=True)
        out_density = torch.sum(x[..., :1], dim=spatial_dims, keepdim=True)
        scaled_density = x[..., :1] * total_density / out_density

        # total_pressure = torch.sum(initial[..., 1:2], dim=spatial_dims, keepdim=True)
        # out_pressure = torch.sum(x[..., 1:2], dim=spatial_dims, keepdim=True)
        scaled_pressure = x[..., 1:2]# * total_pressure / out_pressure
        
        if self.scale_momentum:
            total_momentums = torch.sum(initial[..., :1] * initial[..., 2:], dim=spatial_dims, keepdim=True)
            out_momentums = torch.sum(scaled_density * x[..., 2:], dim=spatial_dims, keepdim=True)
            momentum_ratio = total_momentums / out_momentums
            scaled_velocity = x[..., 2:] * momentum_ratio
            x = torch.concat([scaled_density, scaled_pressure, scaled_velocity], dim=-1)

        elif self.scale_energy:
            total_kinetic_energy = torch.sum(initial[..., :1] * initial[..., 2:] ** 2, dim=spatial_dims, keepdim=True)
            out_kinetic_energy = torch.sum(scaled_density * x[..., 2:] ** 2, dim=spatial_dims, keepdim=True)
            ke_ratio = total_kinetic_energy / out_kinetic_energy
            # v2 = (vx^2 + vy^2 + vz^2) needs to be scaled by ke
            # C*v2 = Cvx^2 + Cvy^2 + Cvz^2
            # C*v2 = (root(C)vx)^2 + ...
            scaled_velocity = x[..., 2:] * torch.sqrt(ke_ratio)
            x = torch.concat([scaled_density, scaled_pressure, scaled_velocity], dim=-1)
                
        else:
            x = torch.concat([scaled_density, scaled_pressure, x[..., 2:]], dim=-1)

        if self.uncertainty:
            x[..., self.nclass // 2:] = x[..., self.nclass // 2:]**2

        return x

    def forward(self, features):
        # Extract gradient features
        if self.grad_features:
            spatial_dims = list(range(1, len(features.shape) - 2))
            grads = [torch.gradient(features, dim=i)[0] for i in spatial_dims] #, spacing=1/(features.shape[i])
            fno_features = torch.concatenate([features, *grads], dim=-1)
        else:
            fno_features = features

        if not self.static:
            fno_features = fno_features.reshape(fno_features.shape[:-2] + (-1,))
        fno_features = fno_features.moveaxis(-1, 1)
        x = self.fno(fno_features)
        x = x.moveaxis(1, -1)

        if not self.static:
            x = x.unsqueeze(-2)

        if self.scale_mass:
            initial = features[..., -1:, -x.shape[-1]:]
            x = self.scale_totals(initial, x)
                
        return x

class PIFNO_grad(PIFNO):
    def __init__(self, *args, **kwargs):
        super(PIFNO_grad, self).__init__(*args, **kwargs)
        
    def scale_grads(self, initial, dx):
        spatial_dims = list(range(1, len(initial.shape) - 2)) # all but the batch and last dim
        
        scaled_density = dx[..., :1] - dx[..., :1].mean(dim=spatial_dims, keepdim=True)
        scaled_pressure = dx[..., 1:2] - dx[..., 1:2].mean(dim=spatial_dims, keepdim=True)
        
        if self.scale_momentum:
            # sum of -grad(rho)*u
            numerator = torch.sum(-scaled_density * initial[..., 2:], dim=spatial_dims, keepdim=True)
            denominator = torch.sum((initial[..., :1] + scaled_density) * dx[..., 2:], dim=spatial_dims, keepdim=True)
            scaled_velocity = dx[..., 2:] * (numerator / denominator)
            dx = torch.concat([scaled_density, scaled_pressure, scaled_velocity], dim=-1)

        elif self.scale_energy:
            raise NotImplementedError('Energy scaling is not implemented yet for gradient models.')
            
        else:
            dx = torch.concat([scaled_density, scaled_pressure, dx[..., 2:]], dim=-1)

        return dx
    
    def forward(self, features):
        fno_features = features
        if not self.static:
            fno_features = fno_features.reshape(fno_features.shape[:-2] + (-1,))
        fno_features = fno_features.moveaxis(-1, 1)
        dx = self.fno(fno_features)
        dx = dx.moveaxis(1, -1)

        if not self.static:
            dx = dx.unsqueeze(-2)
        
        if self.scale_mass:
            initial = features[..., -1:, -dx.shape[-1]:]
            dx = self.scale_grads(initial, dx)

        x = features[..., -1:, -self.nclass:] + dx
        
        return x


class iPIFNO(PIFNO):
    def __init__(self, *args, **kwargs):
        super(iPIFNO, self).__init__(*args, **kwargs)
    
    def scale_totals(self, initial, x):
        spatial_dims = list(range(1, len(initial.shape) - 2)) # all but the batch and last dim
        total_vorticity = torch.sum(initial[..., :1], dim=spatial_dims, keepdim=True)
        
        out_vorticity = torch.sum(x[..., :1], dim=spatial_dims, keepdim=True)
        scaled_vorticity = x[..., :1] * total_vorticity / out_vorticity

        return scaled_vorticity

    def forward(self, features):
        fno_features = features
        if not self.static:
            fno_features = fno_features.reshape(fno_features.shape[:-2] + (-1,))
        fno_features = fno_features.moveaxis(-1, 1)
        x = self.fno(fno_features)
        x = x.transpose(1, -1)

        if not self.static:
            x = x.unsqueeze(-2)        

        if self.scale_mass:
            initial = features[..., -1:, -x.shape[-1]:]
            x = self.scale_totals(initial, x)
        
        return x

class iPIFNO_grad(PIFNO):
    def __init__(self, *args, **kwargs):
        super(iPIFNO_grad, self).__init__(*args, **kwargs)
    
    def scale_grads(self, initial, dx):
        spatial_dims = list(range(1, len(initial.shape) - 2)) # all but the batch and last dim
        scaled_vorticity = dx[..., :1] - dx[..., :1].mean(dim=spatial_dims, keepdim=True)
        return scaled_vorticity

    def forward(self, features):
        fno_features = features
        if not self.static:
            fno_features = fno_features.reshape(fno_features.shape[:-2] + (-1,))
        fno_features = fno_features.moveaxis(-1, 1)
        dx = self.fno(fno_features)
        dx = dx.moveaxis(1, -1)

        if not self.static:
            dx = dx.unsqueeze(-2)        

        if self.scale_mass:
            initial = features[..., -1:, -x.shape[-1]:]
            dx = self.scale_grads(initial, dx)

        x = features[..., -1:, -self.nclass:] + dx
        
        return x


