import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

#============================================================

def get_pde_loss(dims, spacing, vorticity=False):
        if vorticity:
            return partial(pde_iCFD2d_vorticity, spacing=spacing)
        elif dims == 1:
            loss = partial(pde_CFD1d, spacing=spacing)
        elif dims == 2:
            loss = partial(pde_CFD2d, spacing=spacing)
        elif dims == 3:
            loss = partial(pde_CFD3d, spacing=spacing)
        return loss

def pde_CFD1d(y1, y2, spacing, gamma=5/3):
    initial = y1[..., -1:, -y2.shape[-1]:]
    s = torch.concat([initial, y2], dim=-2)

    h = s[..., 0]  # rho
    p = s[..., 1]  # p
    ux = s[..., 2]  # vx
    
    E = p/(gamma - 1.) + 0.5 * h * (ux**2)
    Fx = ux * (E + p)

    # non conservative form
    hu_x = torch.gradient(h * ux, dim=1, spacing=spacing[0])
    
    ux_x, ux_t = torch.gradient(ux, spacing=spacing)
    p_x = torch.gradient(p, dim=1, spacing=spacing[0])

    Fx_x = torch.gradient(Fx, dim=1, spacing=spacing[0])
    h_t = torch.gradient(h, dim=2, spacing=spacing[1])
    E_t = torch.gradient(E, dim=2, spacing=spacing[1])
    
    eq1 = h_t + hu_x
    eq2 = h * (ux_t + ux * ux_x) - p_x
    eq3 = E_t + Fx_x

    residual = (torch.abs(eq1) + torch.abs(eq2) +
                torch.abs(eq3))
    return torch.mean(residual)

def pde_CFD2d(y1, y2, spacing, gamma=5/3):
    initial = y1[..., -1:, -y2.shape[-1]:]
    s = torch.concat([initial, y2], dim=-2)

    h = s[..., 0]  # rho
    p = s[..., 1]  # p
    ux = s[..., 2]  # vx
    uy = s[..., 3]  # vy
    
    E = p/(gamma - 1.) + 0.5 * h * (ux**2 + uy**2)
    Fx = ux * (E + p)
    Fy = uy * (E + p)

    # non conservative form
    hux_x, _ = torch.gradient(h * ux, dim=(1, 2), spacing=spacing[:2])
    _, huy_y = torch.gradient(h * uy, dim=(1, 2), spacing=spacing[:2])

    ux_x, ux_y, ux_t = torch.gradient(ux, dim=(1, 2, 3), spacing=spacing)
    uy_x, uy_y, uy_t = torch.gradient(uy, dim=(1, 2, 3), spacing=spacing)
    p_x, p_y = torch.gradient(p, dim=(1, 2), spacing=spacing[:2])

    Fx_x, _ = torch.gradient(Fx, dim=(1, 2), spacing=spacing[:2])
    _, Fy_y = torch.gradient(Fy, dim=(1, 2), spacing=spacing[:2])
    
    _, _, h_t = torch.gradient(h, dim=(1, 2, 3), spacing=spacing)
    _, _, E_t = torch.gradient(E, dim=(1, 2, 3), spacing=spacing)

    eq1 = h_t + hux_x + huy_y
    eq2 = h * (ux_t + ux * ux_x + uy * ux_y) - p_x
    eq3 = h * (uy_t + ux * uy_x + uy * uy_y) - p_y
    eq4 = E_t + Fx_x + Fy_y

    # residual = (torch.abs(eq1) + torch.abs(eq2) +
    #             torch.abs(eq3) + torch.abs(eq4))
    residual = (torch.abs(eq1) + torch.abs(eq4))
    return torch.mean(residual)

def pde_CFD3d(y1, y2, spacing, gamma=5/3):
    initial = y1[..., -1:, -y2.shape[-1]:]
    s = torch.concat([initial, y2], dim=-2)

    h = s[..., 0]  # rho
    p = s[..., 1]  # p
    ux = s[..., 2]  # vx
    uy = s[..., 3]  # vy
    uz = s[..., 4]  # vz
    
    E = p/(gamma - 1.) + 0.5 * h * (ux**2 + uy**2 + uz**2)
    Fx = ux * (E + p)
    Fy = uy * (E + p)
    Fz = uz * (E + p)

    # non conservative form
    hux_x, _, _ = torch.gradient(h * ux, dim=(1, 2, 3), spacing=spacing[:3])
    _, huy_y, _ = torch.gradient(h * uy, dim=(1, 2, 3), spacing=spacing[:3])
    _, _, huz_z = torch.gradient(h * uz, dim=(1, 2, 3), spacing=spacing[:3])

    ux_x, ux_y, ux_z, ux_t = torch.gradient(ux, dim=(1, 2, 3, 4), spacing=spacing)
    uy_x, uy_y, uy_z, uy_t = torch.gradient(uy, dim=(1, 2, 3, 4), spacing=spacing)
    uz_x, uz_y, uz_z, uz_t = torch.gradient(uz, dim=(1, 2, 3, 4), spacing=spacing)
    p_x, p_y, p_z = torch.gradient(p, dim=(1, 2, 3), spacing=spacing[:3])

    Fx_x, _, _ = torch.gradient(Fx, dim=(1, 2, 3), spacing=spacing[:3])
    _, Fy_y, _ = torch.gradient(Fy, dim=(1, 2, 3), spacing=spacing[:3])
    _, _, Fz_z = torch.gradient(Fz, dim=(1, 2, 3), spacing=spacing[:3])
    
    h_t = torch.gradient(h, dim=(1, 2, 3, 4), spacing=spacing)
    E_t = torch.gradient(E, dim=(1, 2, 3, 4), spacing=spacing)

    eq1 = h_t + hux_x + huy_y + huz_z
    eq2 = h * (ux_t + ux * ux_x + uy * ux_y + uz * ux_z) - p_x
    eq3 = h * (uy_t + ux * uy_x + uy * uy_y + uz * uy_z) - p_y
    eq4 = h * (uz_t + ux * uz_x + uy * uz_y + uz * uz_z) - p_z
    eq5 = E_t + Fx_x + Fy_y + Fz_z

    residual = (torch.abs(eq1) + torch.abs(eq2) + torch.abs(eq3) +
                torch.abs(eq4) + torch.abs(eq5))
    return torch.mean(residual)


#def forcing(S, device):
#    x = torch.tensor(np.linspace(0, 2*np.pi, S, endpoint=False), dtype=torch.float, device=device).reshape(1, S).repeat(S, 1)
#    return -4 * (torch.cos(4*(x))).reshape(1,S,S,1)

def forcing(S, device):
    x1 = torch.tensor(np.linspace(0, 1, S, endpoint=False), dtype=torch.float, device=device).reshape(1, S).repeat(S, 1)
    x2 = torch.tensor(np.linspace(0, 1, S, endpoint=False), dtype=torch.float, device=device).reshape(S, 1).repeat(1, S)
    x = x1 + x2
    return (1.0 * (torch.sin(2 * np.pi * x) + torch.cos(2 * np.pi * x))).reshape(1, S, S, 1)

def pde_iCFD2d_vorticity(w1, w2, v=1e-4, t_interval=1.0):
    w = torch.concat([w1, w2], dim=-2).squeeze(dim=-1)

    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
 
    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    wx = torch.fft.irfft2(wx_h[:, :, :k_max+1], dim=[1,2])
    wy = torch.fft.irfft2(wy_h[:, :, :k_max+1], dim=[1,2])
    wlap = torch.fft.irfft2(wlap_h[:, :, :k_max+1], dim=[1,2])

    dt = nt / (nt-1)
    wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)
    
    residual = wt + (ux*wx + uy*wy - v*wlap)[..., 1:-1] - forcing(nx, device)
    residual = torch.abs(residual[..., -w2.shape[-2]:-1])
    return torch.mean(residual)
