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

class AutoRegressiveModel(DynamicModel):
    def __init__(self, *args, **kwargs):
        super(AutoRegressiveModel, self).__init__(*args, **kwargs)

    def train_step(self, epoch, report_interval):
        self.t_start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        for i, (locations, features, targets) in enumerate(self.dataloader):
            batch_loss = 0
            for ti_target in targets:
                self.optimizer.zero_grad()
                
                output = self.model(locations, features)
                loss_train = self.get_loss(locations, features, ti_target, output)

                loss_train.backward()
                self.optimizer.step()
                dde.grad.clear()

                features = self.concat_timesteps(features, output.detach())
                batch_loss += loss_train

            losses.append(batch_loss.cpu().item())                

            # if (i + 1) % report_interval == 0:
            self.report(locations, features, targets, batch_loss, epoch)
        
        return np.mean(losses)


    def get_loss(self, locations, features, targets, outputs, mask=None):
        lp_loss = self.lp_loss(outputs, targets)
        if self.pde_weight > 0:        
            pde_loss = self.pde_loss(locations, features, outputs)
        else:
            pde_loss = 0
        return lp_loss + self.pde_weight * pde_loss
