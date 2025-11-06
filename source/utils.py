import numpy as np
import torch

def convert_to_conservative_form(data):
    if type(data) is torch.Tensor:
        c_data = data.clone().to(data.device)
    else:
        c_data = data.copy()
    c_data[...,0] = data[...,0]
            
    # momentum
    for i in range(2, data.shape[-1]):
        c_data[...,i] = data[...,i] * data[...,0]

    # pressure last
    v2 = sum(data[...,i]**2 for i in range(2, data.shape[-1]))
    c_data[...,1] = data[...,1] * (1 / (5/3 - 1)) + 0.5 * (v2) * data[...,0]
    return c_data

def convert_from_conservative_form(c_data):
    if type(c_data) is torch.Tensor:
        data = c_data.clone().to(c_data.device)
    else:
        data = c_data.copy()
    data[...,0] = c_data[...,0]
            
    # momentum
    for i in range(2, c_data.shape[-1]):
        data[...,i] = c_data[...,i] / c_data[...,0]

    # pressure last
    v2 = sum(data[...,i]**2 for i in range(2, c_data.shape[-1]))
    data[...,1] = (c_data[...,1] - 0.5 * (v2) * c_data[...,0]) * (5/3 - 1)
    return data