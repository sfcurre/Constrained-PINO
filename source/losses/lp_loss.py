import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

def normalized_lp_loss(x, y, p=2):
    num_examples = x.size()[0]

    diff_norms = torch.linalg.vector_norm(x - y, ord=p, dim=-1)
    y_norms = torch.linalg.vector_norm(y, ord=p, dim=-1)

    return torch.mean(diff_norms/y_norms)

def lp_loss(output, target, dims, weights):
        #loss = normalized_lp_loss(*fargs, **kwargs, p=2)
        output = output * weights
        target = target * weights
        loss = F.mse_loss(output, target)
        dims = list(range(-dims - 1, -1, 1))
        f_output, f_target = torch.fft.rfftn(output, dim=dims, norm='ortho'), torch.fft.rfftn(target, dim=dims, norm='ortho')
        floss = F.mse_loss(f_output.abs(), f_target.abs())
        return loss + floss

def get_lp_loss(dims, weights):
      return partial(lp_loss, dims=dims, weights=weights)

def nll_loss(target, mean, var):
    mse = (mean - target) ** 2
    loss = (torch.log(var) + (mse / var)) / 2
    return loss.mean()