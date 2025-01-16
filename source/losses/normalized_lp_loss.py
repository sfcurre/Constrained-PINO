import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_lp_loss(x, y, p=2):
    num_examples = x.size()[0]

    diff_norms = torch.linalg.vector_norm(x - y, ord=p, dim=-1)
    y_norms = torch.linalg.vector_norm(y, ord=p, dim=-1)

    return torch.mean(diff_norms/y_norms)
