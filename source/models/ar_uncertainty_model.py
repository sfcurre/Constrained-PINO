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

class ARUncertaintyModel(DynamicModel):
    def __init__(self, *args, **kwargs):
        super(ARUncertaintyModel, self).__init__(*args, **kwargs)

    def train_step(self, epoch, report_interval):
        self.t_start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        for i, (features, targets) in enumerate(self.dataloader):
            batch_loss = 0
            initial = features
            
            std = torch.zeros_like(initial)
            if self.use_cuda:
                std = std.cuda()
            
            initial = torch.concat([initial, torch.zeros_like(initial)], dim=-1)

            for i in range(targets.shape[-2]):
                self.optimizer.zero_grad()
                
                output = self.model(initial)
                loss_train = self.get_loss(initial, targets[...,i:i+1,:], output)
                
                initial = self.concat_timesteps(initial, output)
                batch_loss += loss_train

            batch_loss.backward()
            self.optimizer.step()
            
            losses.append(batch_loss.cpu().item())                

            # if (i + 1) % report_interval == 0:
            self.report(features, targets, batch_loss, epoch)
        
        return np.mean(losses)
    
    def mean_std_split(self, outputs):
        num_vars = outputs.shape[-1] // 2
        mean, std = outputs[..., :num_vars], outputs[..., num_vars:]
        return mean, std

    def get_loss(self, features, targets, outputs, mask=None):
        # print(features.shape, outputs.shape, targets.shape)
        mean, std = self.mean_std_split(outputs)
        features_mean, _ = self.mean_std_split(features)
        lp_loss = self.lp_loss(targets, mean, std)
        if self.pde_weight > 0:        
            pde_loss = self.pde_loss(features_mean, mean)
        else:
            pde_loss = 0
        return lp_loss + self.pde_weight * pde_loss
    
    def predict(self, features, timesteps=1, numpy=False):
        # add with torch.no_grad
        outputs = []

        std = torch.zeros_like(features)
        if self.use_cuda:
            std = std.cuda()
        features = torch.concat([features, torch.zeros_like(features)], dim=-1)

        for i in range(timesteps):
            output = self.model(features)
            features = self.concat_timesteps(features, output)
            output = output.detach().cpu()
            if numpy:
                output = output.numpy()
            outputs.append(output)

        if numpy:
            return np.concatenate(outputs, axis=-2)
        return torch.concat(outputs, dim=-2)

    def get_test_loss(self, targets, outputs):
        mean, std = self.mean_std_split(outputs)
        total_loss = self.lp_loss(targets.cpu(), mean.cpu(), std.cpu())
        return total_loss

    def get_test_pde_loss(self, features, outputs):
        mean, std = self.mean_std_split(outputs)
        features_mean, _ = self.mean_std_split(features)
        return self.pde_loss(features_mean.cpu(), mean.cpu())
