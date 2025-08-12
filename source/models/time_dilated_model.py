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

class TimeDilatedModel(DynamicModel):
    def __init__(self, *args, dilation=2, **kwargs):
        super(TimeDilatedModel, self).__init__(*args, **kwargs)
        self.dilation = dilation

    def train_step(self, epoch, report_interval):
        self.t_start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        for i, (features, targets) in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            
            features = self.dilate(features)
            output = self.model(features)
            loss_train = self.get_loss(features, targets, output)
            losses.append(loss_train.cpu().item())
            loss_train.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.scheduler.step()

            #if (i + 1) % report_interval == 0:
            self.report(features, targets, loss_train, epoch)
        
        #self.report(adj, locations, features, targets, mask, loss_train, epoch)    
        return np.mean(losses)
    
    def dilate(self, features):
        if self.dilation > 1:
            dilated_features = torch.repeat_interleave(features, self.dilation, dim=-2)
            return dilated_features
        return features
    
    def contract(self, outputs):
        if self.dilation > 1:
            contracted_outputs = outputs[..., self.dilation//2::self.dilation, :]
            return contracted_outputs
        return outputs

    def get_loss(self, features, targets, outputs):
        contracted_outputs = self.contract(outputs)
        lp_loss = self.lp_loss(contracted_outputs, targets)
        pde_loss = 0
        if self.pde_weight > 0:
            pde_loss = self.pde_loss(features, outputs)
        return (1-self.pde_weight) * lp_loss + self.pde_weight * pde_loss

    def predict(self, features, timesteps=1, numpy=False):
        features = self.dilate(features)
        outputs = self.contract(self.model(features))[:..., :timesteps, :]
        outputs = outputs.detach().cpu()
        if numpy:
            return outputs.numpy()
        return outputs