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

    def train_step(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        for adj, locations, features, targets in self.dataloader:
            
            output_1 = self.model(locations, features, adj)
            output_1 = output_1.detach()
            
            output_2 = self.model(locations, output_1, adj)
            
            loss_train = self.get_loss(locations, output_1, targets[1], output_2)
            
            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        outputs = self.predict(locations, features, adj, timesteps=len(targets))

        loss_val = self.get_test_loss(targets, outputs)
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.data.item()),
            'loss_val: {:.4f}'.format(loss_val.data.item()),
            'time: {:.4f}s'.format(time.time() - t))

        return loss_val.data.item()

    def get_loss(self, locations, features, targets, outputs):
        mse_loss = F.mse_loss(outputs, targets)
        pde_loss = self.pde_loss(locations, features, outputs)
        return mse_loss + pde_loss
