import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NavierStokesLoss2D(nn.Module):
    def __init__(self, rho, nu, continuity_weight=1, momentum_weight=1):
        self.rho_constant = rho
        self.nu_constant = nu
        self.continuity_weight = continuity_weight
        self.momentum_weight = momentum_weight
        super(NavierStokesLoss, self).__init__()
 
    def continuity_loss(self, locations, velocities):
        du_dx = torch.autograd.grad(velocities[0], locations[0])
        dv_dy = torch.autograd.grad(velocities[1], locations[1])
        residual = du_dx + dv_dy
        return residual

    def momentum_loss(self, density, locations, velocities, du_dt):
        du_dx = torch.autograd.grad(velocities[0], locations[0])
        du_dy = torch.autograd.grad(velocities[0], locations[1])
        
        du_dxx = torch.autograd.grad(du_dx, locations[0])
        du_dyy = torch.autograd.grad(du_dy, locations[1])
        
        dp_dx = torch.autograd.grad(density, locations[0])

        residual = (du_dt + velocities[0] * du_dx + velocities[1] * du_dy +
                    dp_dx / self.rho_constant - self.nu_constant * (du_dxx + du_dyy))
        return residual

    def forward(self, inputs, outputs):
        x_i, y_i, p_i, u_i, v_i, E_i = torch.unbind(inputs, dim=-1)
        p_o, u_o, v_o, E_o = torch.unbind(outputs, dim=-1)

        continuity = self.continuity_loss((x_i, y_i), (u_o, v_o, w_o))
        momentum_x = self.momentum_loss(p_o, (x_i, y_i), (u_o, v_o), u_o - u_i)
        momentum_y = self.momentum_loss(p_o, (y_i, x_i), (v_o, u_o), v_o - v_i)
        
        residual = (self.continuity_weight * continuity +
                self.momentum_weight * (momentum_x + momentum_y))

        loss = F.mse_loss(residual, 0)

        return loss


class NavierStokesLoss3D(nn.Module):
    def __init__(self, rho, nu, continuity_weight=1, momentum_weight=1):
        self.rho_constant = rho
        self.nu_constant = nu
        self.continuity_weight = continuity_weight
        self.momentum_weight = momentum_weight
        super(NavierStokesLoss, self).__init__()
 
    def continuity_loss(self, locations, velocities):
        du_dx = torch.autograd.grad(velocities[0], locations[0])
        dv_dy = torch.autograd.grad(velocities[1], locations[1])
        dw_dz = torch.autograd.grad(velocities[2], locaitons[2])
        residual = du_dx + dv_dy + dw_dz
        return residual

    def momentum_loss(self, density, locations, velocities, du_dt):
        du_dx = torch.autograd.grad(velocities[0], locations[0])
        du_dy = torch.autograd.grad(velocities[0], locations[1])
        du_dz = torch.autograd.grad(velocities[0], locations[2])

        du_dxx = torch.autograd.grad(du_dx, locations[0])
        du_dyy = torch.autograd.grad(du_dy, locations[1])
        du_dzz = torch.autograd.grad(du_dz, locations[2])

        dp_dx = torch.autograd.grad(density, locations[0])

        residual = (du_dt + velocities[0] * du_dx + velocities[1] * du_dy +
                    velocities[2] * du_dz + dp_dx / self.rho_constant -
                    self.nu_constant * (du_dxx + du_dyy + du_dzz))
        return residual

    def forward(self, inputs, outputs):
        x_i, y_i, z_i, p_i, u_i, v_i, w_i, E_i = torch.unbind(inputs, dim=-1)
        p_o, u_o, v_o, w_o, E_o = torch.unbind(outputs, dim=-1)

        continuity = self.continuity_loss((x_i, y_i, z_i), (u_o, v_o, w_o))
        momentum_x = self.momentum_loss(p_o, (x_i, y_i, z_i), (u_o, v_o, w_o), u_o - u_i)
        momentum_y = self.momentum_loss(p_o, (y_i, z_i, x_i), (v_o, w_o, u_o), v_o - v_i)
        momentum_z = self.momentum_loss(p_o, (z_i, x_i, y_i), (w_o, u_o, v_o), w_o - w_i)

        residual = (self.continuity_weight * continuity +
                self.momentum_weight * (momentum_x + momentum_y + momentum_z))

        loss = F.mse_loss(residual, 0)

        return loss

class PeriodicNavierStokesLoss3D(nn.Module):
    def __init__(self, rho, nu, continuity_weight=1, momentum_weight=1):
        self.rho_constant = rho
        self.nu_constant = nu
        self.continuity_weight = continuity_weight
        self.momentum_weight = momentum_weight
        super(NavierStokesLoss, self).__init__()
 
    def continuity_loss(self, locations, velocities):
        du_dx = torch.gradient(velocities[0], (locations[0],))
        dv_dy = torch.gradient(velocities[1], (locations[1],))
        dw_dz = torch.gradient(velocities[2], (locations[2],))
        residual = du_dx + dv_dy + dw_dz
        return residual

    def momentum_loss(self, density, locations, velocities, du_dt):   
        du_dx = torch.gradient(velocities[0], (locations[0],))
        du_dy = torch.gradient(velocities[0], (locations[1],))
        du_dz = torch.gradient(velocities[0], (locations[2],))

        du_dxx = torch.autograd.grad(du_dx, locations[0])
        du_dyy = torch.autograd.grad(du_dy, locations[1])
        du_dzz = torch.autograd.grad(du_dz, locations[2])

        dp_dx = torch.gradient(density, (locations[0],))

        residual = (du_dt + velocities[0] * du_dx + velocities[1] * du_dy +
                    velocities[2] * du_dz + dp_dx / self.rho_constant -
                    self.nu_constant * (du_dxx + du_dyy + du_dzz))
        return residual

    def forward(self, inputs, outputs, locations):
        _, _, _, _, _, _, p_i, u_i, v_i, w_i, E_i = torch.unbind(inputs, dim=-1)
        p_o, u_o, v_o, w_o, E_o = torch.unbind(outputs, dim=-1)

        continuity = self.continuity_loss((x_i, y_i, z_i), (u_o, v_o, w_o))
        momentum_x = self.momentum_loss(p_o, (x_i, y_i, z_i), (u_o, v_o, w_o), u_o - u_i)
        momentum_y = self.momentum_loss(p_o, (y_i, z_i, x_i), (v_o, w_o, u_o), v_o - v_i)
        momentum_z = self.momentum_loss(p_o, (z_i, x_i, y_i), (w_o, u_o, v_o), w_o - w_i)

        residual = (self.continuity_weight * continuity +
                self.momentum_weight * (momentum_x + momentum_y + momentum_z))

        loss = F.mse_loss(residual, 0)

        return loss
