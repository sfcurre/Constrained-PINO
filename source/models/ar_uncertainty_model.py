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

class ARUncertaintyModel(DynamicModel):
    def __init__(self, *args, **kwargs):
        super(ARUncertaintyModel, self).__init__(*args, **kwargs)

    def train_step(self, epoch, report_interval):
        self.t_start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        for i, (locations, features, targets) in enumerate(self.dataloader):
            batch_loss = 0
            initial = features
            initial = torch.concat([initial, torch.zeros_like(initial)], dim=-1)

            for i in range(targets.shape[-2]):
                self.optimizer.zero_grad()
                
                output = self.model(locations, initial)
                loss_train = self.get_loss(locations, initial, targets[...,i:i+1,:], output)
                
                initial = self.concat_timesteps(initial, output)
                batch_loss += loss_train

            batch_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            dde.grad.clear()
            
            losses.append(batch_loss.cpu().item())                

            # if (i + 1) % report_interval == 0:
            self.report(locations, features, targets, batch_loss, epoch)
        
        return np.mean(losses)
    
    def mean_var_split(self, outputs):
        num_vars = outputs.shape[-1] // 2
        mean, var = outputs[..., :num_vars], outputs[..., num_vars:]
        return mean, var

    def get_loss(self, locations, features, targets, outputs, mask=None):
        # print(features.shape, outputs.shape, targets.shape)
        mean, var = self.mean_var_split(outputs)
        features_mean, _ = self.mean_var_split(features)
        lp_loss = self.lp_loss(targets, mean, var)
        if self.pde_weight > 0:        
            pde_loss = self.pde_loss(features_mean, mean)
        else:
            pde_loss = 0
        return lp_loss + self.pde_weight * pde_loss
    
    def predict(self, locations, features, timesteps=1, numpy=False):
        # add with torch.no_grad
        outputs = []
        features = torch.concat([features, torch.zeros_like(features)], dim=-1)

        for i in range(timesteps):
            output = self.model(locations, features)
            features = self.concat_timesteps(features, output)
            output = output.detach().cpu()
            if numpy:
                output = output.numpy()
            outputs.append(output)

        if numpy:
            return np.concatenate(outputs, axis=-2)
        return torch.concat(outputs, dim=-2)

    def get_test_loss(self, targets, outputs):
        mean, var = self.mean_var_split(outputs)
        total_loss = self.lp_loss(targets.cpu(), mean.cpu(), var.cpu())
        return total_loss

    def get_test_pde_loss(self, features, outputs):
        mean, var = self.mean_var_split(outputs)
        #features_mean, _ = self.mean_std_split(features)
        return self.pde_loss(features.cpu(), mean.cpu())
