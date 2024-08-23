import os
import glob
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import deepxde as dde

from .dynamic_model import DynamicModel

class OneStepModel(DynamicModel):
    def __init__(self, *args, **kwargs):
        super(OneStepModel, self).__init__(*args, **kwargs)

    def train_step(self, epoch, report_interval):
        self.t_start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        for i, (adj, locations, features, targets, mask) in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            
            output = self.model(locations, features, adj)
            loss_train = self.get_loss(locations, features, targets[0], output, mask)
            losses.append(loss_train.cpu().item())
            loss_train.backward()
            
            self.optimizer.step()            
            self.optimizer.zero_grad()
            dde.grad.clear()

            if (i + 1) % report_interval == 0:
                self.report(adj, locations, features, targets, mask, loss_train, epoch)
        
        return np.mean(losses)


    def get_loss(self, locations, features, targets, outputs, mask=None):
        if mask is not None:
            mse_loss = torch.sum(((outputs - targets) * mask) ** 2) / torch.sum(mask)
        else:
            mse_loss = F.mse_loss(outputs, targets)
            mask = 1
        pde_loss = self.pde_loss(locations, features, outputs, mask)
        return mse_loss + self.pde_weight * pde_loss
