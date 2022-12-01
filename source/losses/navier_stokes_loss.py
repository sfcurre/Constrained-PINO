import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import deepxde as dde

class TestLoss(nn.Module):
    def __init__(self):
        super(TestLoss, self).__init__()
 
    def forward(self, locations, features, outputs):
        residual = outputs - features
        loss = F.mse_loss(residual, torch.ones_like(residual))
        return loss

class NavierStokesLoss2D(nn.Module):
    def __init__(self, nu, continuity_weight=1, momentum_weight=1):
        self.nu_constant = nu
        self.continuity_weight = continuity_weight
        self.momentum_weight = momentum_weight
        super(NavierStokesLoss2D, self).__init__()
 
    def continuity_loss(self, locations, features):
        du_dx = dde.grad.jacobian(features, locations, i=1, j=0)
        dv_dy = dde.grad.jacobian(features, locations, i=2, j=1)

        residual = du_dx + dv_dy
        return residual

    def momentum_loss(self, locations, features, outputs, index):
        di_dx, di_dy = dde.grad.jacobian(outputs, locations, i=index)
        
        di_dxx = dde.grad.hessian(outputs, locations, component=index, i=0, j=0)
        di_dyy = dde.grad.hessian(outputs, locations, component=index, i=1, j=1)

        dp_di = dde.grad.jacobian(outputs, locations, i=0, j=index-1)
        di_dt = (outputs[..., index] - features[..., index]).unsqueeze(-1) 

        u_o = outputs[..., 1].unsqueeze(-1)
        v_o = outputs[..., 2].unsqueeze(-1)
        r_o = outputs[..., -1].unsqueeze(-1)

        residual = (di_dt + u_o * di_dx + v_o * di_dy + 
                    dp_di / r_o - self.nu_constant * (di_dxx + di_dyy))
        return residual

    def forward(self, locations, features, outputs):
        continuity = self.continuity_loss(locations, outputs)
        momentum_x = self.momentum_loss(locations, features, outputs, 1)
        momentum_y = self.momentum_loss(locations, features, outputs, 2)

        residual = (self.continuity_weight * continuity +
                self.momentum_weight * (momentum_x + momentum_y))

        loss = F.mse_loss(residual, torch.zeros_like(residual))

        return loss


class NavierStokesLoss3D(nn.Module):
    def __init__(self, nu, continuity_weight=1, momentum_weight=1):
        self.nu_constant = nu
        self.continuity_weight = continuity_weight
        self.momentum_weight = momentum_weight
        super(NavierStokesLoss2D, self).__init__()
 
    def continuity_loss(self, locations, features):
        du_dx = dde.grad.jacobian(features, locations, i=1, j=0)
        dv_dy = dde.grad.jacobian(features, locations, i=2, j=1)
        dw_dz = dde.grad.jacobian(features, locations, i=3, j=2)

        residual = du_dx + dv_dy + dw_dz
        return residual

    def momentum_loss(self, locations, features, outputs, index):
        di_dx, di_dy, di_dz = dde.grad.jacobian(outputs, locations, i=index)
        
        di_dxx = dde.grad.hessian(outputs, locations, component=index, i=0, j=0)
        di_dyy = dde.grad.hessian(outputs, locations, component=index, i=1, j=1)
        di_dzz = dde.grad.hessian(outputs, locations, component=index, i=2, j=2)

        dp_di = dde.grad.jacobian(outputs, locations, i=0, j=index-1)
        di_dt = (outputs[..., index] - features[..., index]).unsqueeze(-1) 

        u_o = outputs[..., 1].unsqueeze(-1)
        v_o = outputs[..., 2].unsqueeze(-1)
        w_o = outputs[..., 3].unsqueeze(-1)
        r_o = outputs[..., -1].unsqueeze(-1)

        residual = (di_dt + u_o * di_dx + v_o * di_dy + w_o * di_dz +
                    dp_di / r_o - self.nu_constant * (di_dxx + di_dyy + di_dzz))
        return residual

    def forward(self, locations, features, outputs):
        continuity = self.continuity_loss(locations, outputs)
        momentum_x = self.momentum_loss(locations, features, outputs, 1)
        momentum_y = self.momentum_loss(locations, features, outputs, 2)
        momentum_z = self.momentum_loss(locations, features, outputs, 3)

        residual = (self.continuity_weight * continuity +
                self.momentum_weight * (momentum_x + momentum_y + momentum_z))

        loss = F.mse_loss(residual, torch.zeros_like(residual))

        return loss

#=======================================================================

def reaction_1(u1, u2):
    k = 5e-3

    return u1 - (u1 * u1 * u1) - k - u2


def reaction_2(u1, u2):
    return u1 - u2


def pde_diffusion_reaction(x, y):

    d1 = 1e-3
    d2 = 5e-3

    du1_xx = dde.grad.hessian(y, x, i=0, j=0, component=0)
    du1_yy = dde.grad.hessian(y, x, i=1, j=1, component=0)
    du2_xx = dde.grad.hessian(y, x, i=0, j=0, component=1)
    du2_yy = dde.grad.hessian(y, x, i=1, j=1, component=1)

    # TODO: check indices of jacobian
    du1_t = dde.grad.jacobian(y, x, i=0, j=2)
    du2_t = dde.grad.jacobian(y, x, i=1, j=2)

    u1 = y[..., 0].unsqueeze(1)
    u2 = y[..., 1].unsqueeze(1)

    eq1 = du1_t - reaction_1(u1, u2) - d1 * (du1_xx + du1_yy)
    eq2 = du2_t - reaction_2(u1, u2) - d2 * (du2_xx + du2_yy)

    return eq1 + eq2


def pde_diffusion_sorption(x, y):
    D: float = 5e-4
    por: float = 0.29
    rho_s: float = 2880
    k_f: float = 3.5e-4
    n_f: float = 0.874

    du1_xx = dde.grad.hessian(y, x, i=0, j=0)
    # TODO: check indices of jacobian
    du1_t = dde.grad.jacobian(y, x, i=0, j=1)

    u1 = y[..., 0].unsqueeze(1)

    # retardation_factor = 1 + (1 - por) / por * rho_s * k_f * torch.pow(u1, n_f - 1)
    retardation_factor = 1 + ((1 - por) / por) * rho_s * k_f * n_f * (u1 + 1e-6) ** (
        n_f - 1
    )

    return du1_t - D / retardation_factor * du1_xx
    
    
def pde_swe1d():
    raise NotImplementedError


def pde_swe2d(x, y):
    g = 1.0

    # non conservative form
    h_x = dde.grad.jacobian(y, x, i=0, j=0)
    h_y = dde.grad.jacobian(y, x, i=0, j=1)
    h_t = dde.grad.jacobian(y, x, i=0, j=2)
    u_x = dde.grad.jacobian(y, x, i=1, j=0)
    u_y = dde.grad.jacobian(y, x, i=1, j=1)
    u_t = dde.grad.jacobian(y, x, i=1, j=2)
    v_x = dde.grad.jacobian(y, x, i=2, j=0)
    v_y = dde.grad.jacobian(y, x, i=2, j=1)
    v_t = dde.grad.jacobian(y, x, i=2, j=2)

    h = y[..., 0].unsqueeze(1)
    u = y[..., 1].unsqueeze(1)
    v = y[..., 2].unsqueeze(1)

    eq1 = h_t + h_x * u + h * u_x + h_y * v + h * v_y
    eq2 = u_t + u * u_x + v * u_y + g * h_x
    eq3 = v_t + u * v_x + v * v_y + g * h_y

    return eq1 + eq2 + eq3

def pde_adv1d(x, y, beta):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    return dy_t + beta * dy_x

def pde_diffusion_reaction_1d(x, y, nu, rho):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - nu * dy_xx - rho * y * (1. - y)

def pde_burgers1D(x, y, nu):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - nu / np.pi * dy_xx

def pde_CFD1d(x, y, gamma):
    h = y[..., 0].unsqueeze(1)  # rho
    u = y[..., 1].unsqueeze(1)  # v
    p = y[..., 2].unsqueeze(1)  # p
    E = p/(gamma - 1.) + 0.5 * h * u**2
    E = E.unsqueeze(1)
    Fx = u * (E + p)
    Fx = Fx.unsqueeze(1)

    # non conservative form
    hu_x = dde.grad.jacobian(h * u, x, i=0, j=0)
    h_t = dde.grad.jacobian(y, x, i=0, j=1)
    u_x = dde.grad.jacobian(y, x, i=1, j=0)
    u_t = dde.grad.jacobian(y, x, i=1, j=1)
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    Fx_x = dde.grad.jacobian(Fx, x, i=0, j=0)
    E_t = dde.grad.jacobian(E, x, i=0, j=1)

    eq1 = h_t + hu_x
    eq2 = h * (u_t + u * u_x) - p_x
    eq3 = E_t + Fx_x

    return eq1 + eq2 + eq3

def pde_CFD2d(x, y, gamma):
    h = y[..., 0].unsqueeze(1)  # rho
    ux = y[..., 1].unsqueeze(1)  # vx
    uy = y[..., 2].unsqueeze(1)  # vy
    p = y[..., 3].unsqueeze(1)  # p
    E = p/(gamma - 1.) + 0.5 * h * (ux**2 + uy**2)
    E = E.unsqueeze(1)
    Fx = ux * (E + p)
    Fx = Fx.unsqueeze(1)
    Fy = uy * (E + p)
    Fy = Fy.unsqueeze(1)

    # non conservative form
    hu_x = dde.grad.jacobian(h * ux, x, i=0, j=0)
    hu_y = dde.grad.jacobian(h * uy, x, i=0, j=1)
    h_t = dde.grad.jacobian(y, x, i=0, j=2)
    ux_x = dde.grad.jacobian(y, x, i=1, j=0)
    ux_y = dde.grad.jacobian(y, x, i=1, j=1)
    ux_t = dde.grad.jacobian(y, x, i=1, j=2)
    uy_x = dde.grad.jacobian(y, x, i=2, j=0)
    uy_y = dde.grad.jacobian(y, x, i=2, j=1)
    uy_t = dde.grad.jacobian(y, x, i=2, j=2)
    p_x = dde.grad.jacobian(y, x, i=3, j=0)
    p_y = dde.grad.jacobian(y, x, i=3, j=1)
    Fx_x = dde.grad.jacobian(Fx, x, i=0, j=0)
    Fy_y = dde.grad.jacobian(Fy, x, i=0, j=1)
    E_t = dde.grad.jacobian(E, x, i=0, j=2)

    eq1 = h_t + hu_x + hu_y
    eq2 = h * (ux_t + ux * ux_x + uy * ux_y) - p_x
    eq3 = h * (uy_t + ux * uy_x + uy * uy_y) - p_y
    eq4 = E_t + Fx_x + Fy_y

    return eq1 + eq2 + eq3 + eq4

def pde_CFD3d(x, y, gamma):
    h = y[..., 0].unsqueeze(1)  # rho
    ux = y[..., 1].unsqueeze(1)  # vx
    uy = y[..., 2].unsqueeze(1)  # vy
    uz = y[..., 3].unsqueeze(1)  # vz
    p = y[..., 4].unsqueeze(1)  # p
    E = p/(gamma - 1.) + 0.5 * h * (ux**2 + uy**2 + uz**2)
    E = E.unsqueeze(1)
    Fx = ux * (E + p)
    Fx = Fx.unsqueeze(1)
    Fy = uy * (E + p)
    Fy = Fy.unsqueeze(1)
    Fz = uz * (E + p)
    Fz = Fz.unsqueeze(1)

    # non conservative form
    hu_x = dde.grad.jacobian(h * ux, x, i=0, j=0)
    hu_y = dde.grad.jacobian(h * uy, x, i=0, j=1)
    hu_z = dde.grad.jacobian(h * uy, x, i=0, j=2)
    h_t = dde.grad.jacobian(y, x, i=0, j=3)
    ux_x = dde.grad.jacobian(y, x, i=1, j=0)
    ux_y = dde.grad.jacobian(y, x, i=1, j=1)
    ux_z = dde.grad.jacobian(y, x, i=1, j=2)
    ux_t = dde.grad.jacobian(y, x, i=1, j=3)
    uy_x = dde.grad.jacobian(y, x, i=2, j=0)
    uy_y = dde.grad.jacobian(y, x, i=2, j=1)
    uy_z = dde.grad.jacobian(y, x, i=2, j=2)
    uy_t = dde.grad.jacobian(y, x, i=2, j=3)
    uz_x = dde.grad.jacobian(y, x, i=3, j=0)
    uz_y = dde.grad.jacobian(y, x, i=3, j=1)
    uz_z = dde.grad.jacobian(y, x, i=3, j=2)
    uz_t = dde.grad.jacobian(y, x, i=3, j=3)
    p_x = dde.grad.jacobian(y, x, i=4, j=0)
    p_y = dde.grad.jacobian(y, x, i=4, j=1)
    p_z = dde.grad.jacobian(y, x, i=4, j=2)
    Fx_x = dde.grad.jacobian(Fx, x, i=0, j=0)
    Fy_y = dde.grad.jacobian(Fy, x, i=0, j=1)
    Fz_z = dde.grad.jacobian(Fz, x, i=0, j=2)
    E_t = dde.grad.jacobian(E, x, i=0, j=3)

    eq1 = h_t + hu_x + hu_y + hu_z
    eq2 = h * (ux_t + ux * ux_x + uy * ux_y + uz * ux_z) - p_x
    eq3 = h * (uy_t + ux * uy_x + uy * uy_y + uz * uy_z) - p_y
    eq4 = h * (uz_t + ux * uz_x + uy * uz_y + uz * uz_z) - p_z
    eq5 = E_t + Fx_x + Fy_y + Fz_z

    return eq1 + eq2 + eq3 + eq4 + eq5