import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import deepxde as dde

#============================================================

def pde_CFD1d(x, y1, y2, mask, gamma=1.4):
    h1 = y1[..., -3].squeeze()  # rho
    ux1 = y1[..., -2].squeeze()  # vx
    p1 = y1[..., -1].squeeze()  # p
    E1 = p1/(gamma - 1.) + 0.5 * h1 * (ux1**2)
    E1 = E1.squeeze()
    
    h2 = y2[..., -3].squeeze()  # rho
    ux2 = y2[..., -2].squeeze()  # vx
    p2 = y2[..., -1].squeeze()  # p
    E2 = p2/(gamma - 1.) + 0.5 * h2 * (ux2**2)
    E2 = E2.squeeze()
    Fx = ux2 * (E2 + p2)
    Fx = Fx.squeeze()
    
    # non conservative form
    hu_x = dde.grad.jacobian(h2 * ux2, x, i=0, j=0)
    
    ux_x = dde.grad.jacobian(y2, x, i=-2, j=0)
    p_x = dde.grad.jacobian(y2, x, i=-1, j=0)
    
    Fx_x = dde.grad.jacobian(Fx, x, i=0, j=0)
    
    h_t = h2 - h1
    ux_t = ux2 - ux1
    E_t = E2 - E1

    eq1 = h_t + hu_x
    eq2 = h2 * (ux_t + ux2 * ux_x) - p_x
    eq3 = E_t + Fx_x

    residual = (torch.abs(eq1) + torch.abs(eq2) +
                torch.abs(eq3))
    loss = residual * mask
    return torch.mean(loss)

def pde_CFD1d_approx(x, y1, y2, mask, gamma=1.4):
    h1 = y1[..., -3].squeeze()  # rho
    ux1 = y1[..., -2].squeeze()  # vx
    p1 = y1[..., -1].squeeze()  # p
    E1 = p1/(gamma - 1.) + 0.5 * h1 * (ux1**2)
    E1 = E1.squeeze()
    
    h2 = y2[..., -3].squeeze()  # rho
    ux2 = y2[..., -2].squeeze()  # vx
    p2 = y2[..., -1].squeeze()  # p
    E2 = p2/(gamma - 1.) + 0.5 * h2 * (ux2**2)
    E2 = E2.squeeze()
    Fx = ux2 * (E2 + p2)
    Fx = Fx.squeeze()
    
    # non conservative form

    hu_x, = torch.gradient(h2 * ux2)
    
    ux_x, = torch.gradient(ux2)
    p_x, = torch.gradient(p2)

    Fx_x, = torch.gradient(Fx)
    
    h_t = h2 - h1
    ux_t = ux2 - ux1
    E_t = E2 - E1

    eq1 = h_t + hu_x
    eq2 = h2 * (ux_t + ux2 * ux_x) - p_x
    eq3 = E_t + Fx_x

    residual = (torch.abs(eq1) + torch.abs(eq2) +
                torch.abs(eq3))
    loss = residual * mask
    return torch.mean(loss)

def pde_CFD2d(x, y1, y2, mask, gamma=1.4):
    h1 = y1[..., -4].squeeze()  # rho
    ux1 = y1[..., -3].squeeze()  # vx
    uy1 = y1[..., -2].squeeze()  # vy
    p1 = y1[..., -1].squeeze()  # p
    E1 = p1/(gamma - 1.) + 0.5 * h1 * (ux1**2 + uy1**2)
    E1 = E1.squeeze()
    
    h2 = y2[..., -4].squeeze()  # rho
    ux2 = y2[..., -3].squeeze()  # vx
    uy2 = y2[..., -2].squeeze()  # vy
    p2 = y2[..., -1].squeeze()  # p
    E2 = p2/(gamma - 1.) + 0.5 * h2 * (ux2**2 + uy2**2)
    E2 = E2.squeeze()
    Fx = ux2 * (E2 + p2)
    Fx = Fx.squeeze()
    Fy = uy2 * (E2 + p2)
    Fy = Fy.squeeze()

    # non conservative form
    hu_x = dde.grad.jacobian(h2 * ux2, x, i=0, j=0)
    hu_y = dde.grad.jacobian(h2 * uy2, x, i=0, j=1)
    
    ux_x = dde.grad.jacobian(y2, x, i=-3, j=0)
    ux_y = dde.grad.jacobian(y2, x, i=-3, j=1)
    
    uy_x = dde.grad.jacobian(y2, x, i=-2, j=0)
    uy_y = dde.grad.jacobian(y2, x, i=-2, j=1)
    
    p_x = dde.grad.jacobian(y2, x, i=-1, j=0)
    p_y = dde.grad.jacobian(y2, x, i=-1, j=1)
    
    Fx_x = dde.grad.jacobian(Fx, x, i=0, j=0)
    Fy_y = dde.grad.jacobian(Fy, x, i=0, j=1)
    
    h_t = h2 - h1
    ux_t = ux2 - ux1
    uy_t = uy2 - uy1
    E_t = E2 - E1

    eq1 = h_t + hu_x + hu_y
    eq2 = h2 * (ux_t + ux2 * ux_x + uy2 * ux_y) - p_x
    eq3 = h2 * (uy_t + ux2 * uy_x + uy2 * uy_y) - p_y
    eq4 = E_t + Fx_x + Fy_y

    residual = (torch.abs(eq1) + torch.abs(eq2) +
                torch.abs(eq3) + torch.abs(eq4))
    loss = residual * mask
    return torch.mean(loss)

def pde_CFD2d_approx(x, y1, y2, mask, gamma=1.4):
    h1 = y1[..., -4].squeeze()  # rho
    ux1 = y1[..., -3].squeeze()  # vx
    uy1 = y1[..., -2].squeeze()  # vy
    p1 = y1[..., -1].squeeze()  # p
    E1 = p1/(gamma - 1.) + 0.5 * h1 * (ux1**2 + uy1**2)
    E1 = E1.squeeze()
    
    h2 = y2[..., -4].squeeze()  # rho
    ux2 = y2[..., -3].squeeze()  # vx
    uy2 = y2[..., -2].squeeze()  # vy
    p2 = y2[..., -1].squeeze()  # p
    E2 = p2/(gamma - 1.) + 0.5 * h2 * (ux2**2 + uy2**2)
    E2 = E2.squeeze()
    Fx = ux2 * (E2 + p2)
    Fx = Fx.squeeze()
    Fy = uy2 * (E2 + p2)
    Fy = Fy.squeeze()

    # non conservative form

    hu_x, _ = torch.gradient(h2 * ux2)
    _, hu_y = torch.gradient(h2 * uy2)

    ux_x, ux_y = torch.gradient(ux2, dim=(0, 1))
    uy_x, uy_y = torch.gradient(uy2, dim=(0, 1))
    p_x, p_y = torch.gradient(p2, dim=(0, 1))

    Fx_x, _ = torch.gradient(Fx)
    _, Fy_y = torch.gradient(Fy)
    
    h_t = h2 - h1
    ux_t = ux2 - ux1
    uy_t = uy2 - uy1
    E_t = E2 - E1

    eq1 = h_t + hu_x + hu_y
    eq2 = h2 * (ux_t + ux2 * ux_x + uy2 * ux_y) - p_x
    eq3 = h2 * (uy_t + ux2 * uy_x + uy2 * uy_y) - p_y
    eq4 = E_t + Fx_x + Fy_y

    residual = (torch.abs(eq1) + torch.abs(eq2) +
                torch.abs(eq3) + torch.abs(eq4))
    loss = residual * mask
    return torch.mean(loss)


def pde_CFD3d(x, y1, y2, mask, gamma=1.4):
    h1 = y1[..., -5].squeeze()  # rho
    ux1 = y1[..., -4].squeeze()  # vx
    uy1 = y1[..., -3].squeeze()  # vy
    uz1 = y1[..., -2].squeeze()  # vz
    p1 = y1[..., -1].squeeze()  # p
    E1 = p1/(gamma - 1.) + 0.5 * h1 * (ux1**2 + uy1**2 + uz1**2)
    E1 = E1.squeeze()
    
    h2 = y2[..., -5].squeeze()  # rho
    ux2 = y2[..., -4].squeeze()  # vx
    uy2 = y2[..., -3].squeeze()  # vy
    uz2 = y2[..., -2].squeeze()  # vz
    p2 = y2[..., -1].squeeze()  # p
    E2 = p2/(gamma - 1.) + 0.5 * h2 * (ux2**2 + uy2**2 + uz2**2)
    E2 = E2.squeeze()
    Fx = ux2 * (E2 + p2)
    Fx = Fx.squeeze()
    Fy = uy2 * (E2 + p2)
    Fy = Fy.squeeze()
    Fz = uz2 * (E2 + p2)
    Fz = Fz.squeeze()

    # non conservative form
    hu_x = dde.grad.jacobian(h2 * ux2, x, i=0, j=0)
    hu_y = dde.grad.jacobian(h2 * uy2, x, i=0, j=1)
    hu_z = dde.grad.jacobian(h2 * uz2, x, i=0, j=2)

    ux_x = dde.grad.jacobian(y2, x, i=-4, j=0)
    ux_y = dde.grad.jacobian(y2, x, i=-4, j=1)
    ux_z = dde.grad.jacobian(y2, x, i=-4, j=2)

    uy_x = dde.grad.jacobian(y2, x, i=-3, j=0)
    uy_y = dde.grad.jacobian(y2, x, i=-3, j=1)
    uy_z = dde.grad.jacobian(y2, x, i=-3, j=2)

    uz_x = dde.grad.jacobian(y2, x, i=-2, j=0)
    uz_y = dde.grad.jacobian(y2, x, i=-2, j=1)
    uz_z = dde.grad.jacobian(y2, x, i=-2, j=2)

    p_x = dde.grad.jacobian(y2, x, i=-1, j=0)
    p_y = dde.grad.jacobian(y2, x, i=-1, j=1)
    p_z = dde.grad.jacobian(y2, x, i=-1, j=2)

    Fx_x = dde.grad.jacobian(Fx, x, i=0, j=0)
    Fy_y = dde.grad.jacobian(Fy, x, i=0, j=1)
    Fz_z = dde.grad.jacobian(Fz, x, i=0, j=2)
    
    h_t = h2 - h1
    ux_t = ux2 - ux1
    uy_t = uy2 - uy1
    uz_t = uz2 - uz1
    E_t = E2 - E1

    eq1 = h_t + hu_x + hu_y + hu_z
    eq2 = h2 * (ux_t + ux2 * ux_x + uy2 * ux_y + uz2 * ux_z) - p_x
    eq3 = h2 * (uy_t + ux2 * uy_x + uy2 * uy_y + uz2 * uy_z) - p_y
    eq4 = h2 * (uz_t + ux2 * uz_x + uy2 * uz_y + uz2 * uz_z) - p_z
    eq5 = E_t + Fx_x + Fy_y + Fz_z

    residual = (torch.abs(eq1) + torch.abs(eq2) + torch.abs(eq3) +
                torch.abs(eq4) + torch.abs(eq5))
    loss = residual * mask
    return torch.mean(loss)


def pde_CFD3d_approx(x, y1, y2, mask, gamma=1.4):
    h1 = y1[..., -5].squeeze()  # rho
    ux1 = y1[..., -4].squeeze()  # vx
    uy1 = y1[..., -3].squeeze()  # vy
    uz1 = y1[..., -2].squeeze()  # vz
    p1 = y1[..., -1].squeeze()  # p
    E1 = p1/(gamma - 1.) + 0.5 * h1 * (ux1**2 + uy1**2 + uz1**2)
    E1 = E1.squeeze()
    
    h2 = y2[..., -5].squeeze()  # rho
    ux2 = y2[..., -4].squeeze()  # vx
    uy2 = y2[..., -3].squeeze()  # vy
    uz2 = y2[..., -2].squeeze()  # vz
    p2 = y2[..., -1].squeeze()  # p
    E2 = p2/(gamma - 1.) + 0.5 * h2 * (ux2**2 + uy2**2 + uz2**2)
    E2 = E2.squeeze()
    Fx = ux2 * (E2 + p2)
    Fx = Fx.squeeze()
    Fy = uy2 * (E2 + p2)
    Fy = Fy.squeeze()
    Fz = uz2 * (E2 + p2)
    Fz = Fz.squeeze()

    # non conservative form
    hu_x, _, _ = torch.gradient(h2 * ux2)
    _, hu_y, _ = torch.gradient(h2 * uy2)
    _, _, hu_z = torch.gradient(h2 * uz2)

    ux_x, ux_y, ux_z = torch.gradient(ux2, dim=(0, 1, 2))
    uy_x, uy_y, uy_z = torch.gradient(uy2, dim=(0, 1, 2))
    uz_x, uz_y, uz_z = torch.gradient(uz2, dim=(0, 1, 2))
    p_x, p_y, p_z = torch.gradient(p2, dim=(0, 1, 2))

    Fx_x, _, _ = torch.gradient(Fx)
    _, Fy_y, _ = torch.gradient(Fy)
    _, _, Fz_z = torch.gradient(Fz)
    
    h_t = h2 - h1
    ux_t = ux2 - ux1
    uy_t = uy2 - uy1
    uz_t = uz2 - uz1
    E_t = E2 - E1

    eq1 = h_t + hu_x + hu_y + hu_z
    eq2 = h2 * (ux_t + ux2 * ux_x + uy2 * ux_y + uz2 * ux_z) - p_x
    eq3 = h2 * (uy_t + ux2 * uy_x + uy2 * uy_y + uz2 * uy_z) - p_y
    eq4 = h2 * (uz_t + ux2 * uz_x + uy2 * uz_y + uz2 * uz_z) - p_z
    eq5 = E_t + Fx_x + Fy_y + Fz_z

    residual = (torch.abs(eq1) + torch.abs(eq2) + torch.abs(eq3) +
                torch.abs(eq4) + torch.abs(eq5))
    loss = residual * mask
    return torch.mean(loss)

