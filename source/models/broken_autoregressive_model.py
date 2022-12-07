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

class BrokenAutoRegressiveModel(DynamicModel):
    def __init__(self, *args, **kwargs):
        super(BrokenAutoRegressiveModel, self).__init__(*args, **kwargs)

    def train_step(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        for adj, locations, features, targets, mask in self.dataloader:
            self.optimizer.zero_grad()
            
            outputs = []
            tmp_feats = features
            for ti_target in targets:
                output = self.model(locations, tmp_feats, adj)
                outputs.append(output)
                tmp_feats = output.detach()

            loss_train = self.get_loss(locations, features, targets, outputs, mask)
            
            loss_train.backward(retain_graph=True)
            self.optimizer.step()


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
        total_loss = 0
        for output, target in zip(outputs, targets):
            if mask is not None:
                mse_loss = torch.sum(((output - target) * mask) ** 2) / torch.sum(mask)
            else:
                mse_loss = F.mse_loss(output, target)
            pde_loss = self.pde_loss(locations, features, output, mask)
            total_loss += mse_loss + pde_loss
            features = output
        return total_loss
