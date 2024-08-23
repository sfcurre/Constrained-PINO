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

class PushForwardModel(DynamicModel):
    def __init__(self, *args, **kwargs):
        super(PushForwardModel, self).__init__(*args, **kwargs)

    def train_step(self, epoch, report_interval):
        self.t_start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        for i, (adj, locations, features, targets, mask) in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            output_1 = self.model(locations, features, adj)
            output_1 = self.concat_timesteps(features, output_1.detach())
            
            output_2 = self.model(locations, output_1, adj)
            
            loss_train = self.get_loss(locations, output_1, targets[1], output_2, mask)
            
            loss_train.backward()
            self.optimizer.step()
            dde.grad.clear()

            if (i + 1) % report_interval == 0:
                self.report(adj, locations, features, targets, mask, loss_train, epoch)
        
        return self.report(adj, locations, features, targets, mask, loss_train, epoch)


    def get_loss(self, locations, features, targets, outputs, mask=None):
        if mask is not None:
            mse_loss = torch.sum(((outputs - targets) * mask) ** 2) / torch.sum(mask)
        else:
            mse_loss = F.mse_loss(outputs, targets)
            mask = 1
        pde_loss = self.pde_loss(locations, features, outputs, mask)
        return mse_loss + self.pde_weight * pde_loss
