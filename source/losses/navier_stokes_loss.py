import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NavierStokesLoss2D(nn.Module):
    def __init__(self, nu, continuity_weight=1, momentum_weight=1):
        self.nu_constant = nu
        self.continuity_weight = continuity_weight
        self.momentum_weight = momentum_weight
        super(NavierStokesLoss2D, self).__init__()
 
    def continuity_loss(self, locations, velocities):
        du_dx = torch.autograd.grad(velocities[0], locations[0])
        dv_dy = torch.autograd.grad(velocities[1], locations[1])
        residual = du_dx + dv_dy
        return residual

    def momentum_loss(self, density, locations, velocities, du_dt, rho):
        du_dx = torch.autograd.grad(velocities[0], locations[0])
        du_dy = torch.autograd.grad(velocities[0], locations[1])
        
        du_dxx = torch.autograd.grad(du_dx, locations[0])
        du_dyy = torch.autograd.grad(du_dy, locations[1])
        
        dp_dx = torch.autograd.grad(density, locations[0])

        residual = (du_dt + velocities[0] * du_dx + velocities[1] * du_dy +
                    dp_dx / rho - self.nu_constant * (du_dxx + du_dyy))
        return residual

    def forward(self, inputs, outputs):
        x_i, y_i, p_i, u_i, v_i, E_i = torch.unbind(inputs, dim=-1)
        p_o, u_o, v_o, r_o = torch.unbind(outputs, dim=-1)

        continuity = self.continuity_loss((x_i, y_i), (u_o, v_o, w_o))
        momentum_x = self.momentum_loss(p_o, (x_i, y_i), (u_o, v_o), u_o - u_i, r_o)
        momentum_y = self.momentum_loss(p_o, (y_i, x_i), (v_o, u_o), v_o - v_i, r_o)
        
        residual = (self.continuity_weight * continuity +
                self.momentum_weight * (momentum_x + momentum_y))

        loss = F.mse_loss(residual, 0)

        return loss

class NavierStokesLoss3D(nn.Module):
    def __init__(self, nu, continuity_weight=1, momentum_weight=1):
        self.nu_constant = nu
        self.continuity_weight = continuity_weight
        self.momentum_weight = momentum_weight
        super(NavierStokesLoss3D, self).__init__()
 
    def continuity_loss(self, locations, velocities):
        du_dx, _, _ = torch.autograd.grad(velocities[0].sum(), locations, 
                                          retain_graph=True)[0].T
        _, dv_dy, _ = torch.autograd.grad(velocities[1].sum(), locations, 
                                          retain_graph=True)[0].T
        _, _, dw_dz = torch.autograd.grad(velocities[2].sum(), locations, 
                                          retain_graph=True)[0].T
        residual = du_dx + dv_dy + dw_dz
        return residual

    def momentum_loss(self, density, locations, velocities, di_dt, rho, index):
        di_dx, di_dy, di_dz = torch.autograd.grad(velocities[index].sum(),
        locations, create_graph=True, retain_graph=True)[0].T
        
        di_dxx, _, _ = torch.autograd.grad(di_dx.sum(), locations, 
                                           retain_graph=True)[0].T
        _, di_dyy, _ = torch.autograd.grad(di_dy.sum(), locations, 
                                           retain_graph=True)[0].T
        _, _, di_dzz = torch.autograd.grad(di_dz.sum(), locations, 
                                           retain_graph=True)[0].T

        dp_di = torch.autograd.grad(density.sum(), locations, 
                                    retain_graph=True)[0][:, index]

        residual = (di_dt + velocities[0] * di_dx + velocities[1] * di_dy +
                    velocities[2] * di_dz + dp_di / rho -
                    self.nu_constant * (di_dxx + di_dyy + di_dzz))
        return residual

    def forward(self, locations, features, outputs):
        p_i, u_i, v_i, w_i, r_i = torch.unbind(features, dim=-1)
        p_o, u_o, v_o, w_o, r_o = torch.unbind(outputs, dim=-1)

        continuity = self.continuity_loss(locations, (u_o, v_o, w_o))
        momentum_x = self.momentum_loss(p_o, locations, (u_o, v_o, w_o), u_o - u_i, r_o, 0)
        momentum_y = self.momentum_loss(p_o, locations, (u_o, v_o, w_o), v_o - v_i, r_o, 1)
        momentum_z = self.momentum_loss(p_o, locations, (u_o, v_o, w_o), w_o - w_i, r_o, 2)

        residual = (self.continuity_weight * continuity +
                self.momentum_weight * (momentum_x + momentum_y + momentum_z))

        loss = F.mse_loss(residual, torch.zeros_like(residual))

        return loss
