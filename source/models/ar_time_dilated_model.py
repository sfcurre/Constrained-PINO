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

class ARTimeDilatedModel(DynamicModel):
    def __init__(self, *args, dilation=2, **kwargs):
        super(ARTimeDilatedModel, self).__init__(*args, **kwargs)
        self.dilation = dilation

    def train_step(self, epoch, report_interval):
        self.t_start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        for i, (features, targets) in enumerate(self.dataloader):
            dilated_features = self.dilate(features)
            initial = dilated_features
            outputs = []
            self.optimizer.zero_grad()        
            for i in range(targets.shape[-2]):
                for j in range(self.dilation):
                    output = self.model(initial)
                    initial = self.concat_timesteps(initial, output.detach())
                    outputs.append(output)

            loss = self.get_loss(features, targets, torch.concat(outputs, dim=-2))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            losses.append(loss.cpu().item())                

            #if (i + 1) % report_interval == 0:
            self.report(features, targets, loss, epoch)
        
        #self.report(adj, locations, features, targets, mask, loss_train, epoch)    
        return np.mean(losses)
    
    def dilate(self, features):
        if self.dilation > 1:
            # dilated_features = torch.repeat_interleave(features, self.dilation, dim=-2)
            # dilated_len = dilated_features.shape[-2] - self.dilation + 1
            # dilated_features = torch.stack([dilated_features[i:i+dilated_len] for i in range(self.dilation)], dim=-1)
            # dilated_features = torch.mean(dilated_features, dim=-1)
            dilated_features = features.transpose(-1, -2).view((-1, features.shape[-1], features.shape[-2]))
            dilated_features = F.interpolate(dilated_features, size=self.dilation * features.shape[-2] - 1, mode='linear', align_corners=True)
            dilated_features = dilated_features.transpose(-1, -2).view(*features.shape[:-2], -1, features.shape[-1])
            return dilated_features
        return features
    
    def contract(self, outputs):
        if self.dilation > 1:
            contracted_outputs = outputs[..., self.dilation-1::self.dilation, :]
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
        outputs = []
        for i in range(timesteps):
            for j in range(self.dilation):
                output = self.model(features)
                features = self.concat_timesteps(features, output.detach())
            
            output = output.detach().cpu()
            if numpy:
                output = output.numpy()
            outputs.append(output)

        if numpy:
            return np.concatenate(outputs, axis=-2)
        return torch.concat(outputs, dim=-2)
