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

class PushForwardModel(DynamicModel):
    def __init__(self, *args, **kwargs):
        super(PushForwardModel, self).__init__(*args, **kwargs)

    def train_step(self, epoch, report_interval):
        self.t_start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        for i, (features, targets) in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            output_1 = self.model(features)
            output_1 = self.concat_timesteps(features, output_1.detach())
            
            output_2 = self.model(output_1)
            
            loss_train = self.get_loss(output_1, targets[..., 1:2, :], output_2)
            
            loss_train.backward()
            self.optimizer.step()

            if (i + 1) % report_interval == 0:
                self.report(features, targets, loss_train, epoch)
        
        return self.report(features, targets, loss_train, epoch)


    def get_loss(self, features, targets, outputs):
        lp_loss = self.lp_loss(outputs, targets)
        if self.pde_weight > 0:        
            pde_loss = self.pde_loss(features, outputs)
        else:
            pde_loss = 0
        return lp_loss + self.pde_weight * pde_loss
