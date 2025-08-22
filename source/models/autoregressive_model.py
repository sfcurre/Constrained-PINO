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

from .dynamic_model import DynamicModel

class AutoRegressiveModel(DynamicModel):
    def __init__(self, *args, **kwargs):
        super(AutoRegressiveModel, self).__init__(*args, **kwargs)

    def train_step(self, epoch, report_interval):
        self.t_start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        for _, (features, targets) in enumerate(self.dataloader):
            initial = features
            outputs = []
            self.optimizer.zero_grad()
            for i in range(targets.shape[-2]):    
                output = self.model(initial)
                initial = self.concat_timesteps(initial, output.detach())
                outputs.append(output)

            loss = self.get_loss(features, targets, torch.concat(outputs, dim=-2))    

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            losses.append(loss.cpu().item())                

            if (i + 1) % report_interval == 0:
                self.report(features, targets, loss, epoch)
        
        return np.mean(losses)

    def get_loss(self, features, targets, outputs, mask=None):
        # print(features.shape, outputs.shape, targets.shape)
        lp_loss = self.lp_loss(outputs, targets)
        pde_loss = 0
        if self.pde_weight > 0:
            pde_loss = self.pde_loss(features, outputs)
        return (1-self.pde_weight) * lp_loss + self.pde_weight * pde_loss
        