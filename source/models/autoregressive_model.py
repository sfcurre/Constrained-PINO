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
from pytorch_memlab import profile

from .dynamic_model import DynamicModel

class AutoRegressiveModel(DynamicModel):
    def __init__(self, *args, **kwargs):
        super(AutoRegressiveModel, self).__init__(*args, **kwargs)

    @profile
    def train_step(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        for adj, locations, features, targets, mask in self.dataloader:
            for ti_target in targets:
                self.optimizer.zero_grad()
                
                output = self.model(locations, features, adj)
                loss_train = self.get_loss(locations, features, ti_target, output, mask)
                
                loss_train.backward(retain_graph=True)
                self.optimizer.step()

                features = output.detach()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        outputs = self.predict(locations, features, adj, timesteps=len(targets))

        loss_val = self.get_test_loss(targets, outputs, mask)
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.data.item()),
            'loss_val: {:.4f}'.format(loss_val.data.item()),
            'time: {:.4f}s'.format(time.time() - t))

        return loss_val.data.item()

    def get_loss(self, locations, features, targets, outputs, mask=None):
        if mask is not None:
            mse_loss = torch.sum(((outputs - targets) * mask) ** 2) / torch.sum(mask)
        else:
            mse_loss = F.mse_loss(outputs, targets)
        pde_loss = self.pde_loss(locations, features, outputs, mask)
        return mse_loss + pde_loss
