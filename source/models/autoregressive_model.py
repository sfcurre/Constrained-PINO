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
import matplotlib.pyplot as plt

class AutoRegressiveModel(DynamicModel):
    def __init__(self, *args, **kwargs):
        super(AutoRegressiveModel, self).__init__(*args, **kwargs)

    def train_step(self, epoch, report_interval):
        self.t_start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        for i, (features, targets) in enumerate(self.dataloader):
            initial = features
            
            # plt.imshow(initial[0, ..., -1, 0].detach().numpy())
            # plt.show()

            # plt.imshow(targets[0, ..., 0, 0].detach().numpy())
            # plt.show()

            outputs = []
            self.optimizer.zero_grad()
            for _ in range(targets.shape[-2]):    
                output = self.model(initial)
                # plt.imshow(output[0, ..., 0, 0].detach().numpy())
                # plt.show()

                initial = self.concat_timesteps(initial, output)
                outputs.append(output)

            loss = self.get_loss(features, targets, torch.concat(outputs, dim=-2))

            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.cpu().item())

            if (i + 1) % report_interval == 0:
                self.report(features, targets, loss, epoch)
        
        self.report(features, targets, loss, epoch)
        return np.mean(losses)

    def get_loss(self, features, targets, outputs, mask=None):
        losses = []
        for i in range(targets.shape[-2]):
            output = outputs[..., i:i+1, :]
            target = targets[..., i:i+1, :]
            losses.append(self.lp_loss(output, target))
        lp_loss = torch.mean(torch.stack(losses))
        pde_loss = 0
        if self.pde_weight > 0:
            pde_loss = self.pde_loss(features, outputs)
        return (1-self.pde_weight) * lp_loss + self.pde_weight * pde_loss
        