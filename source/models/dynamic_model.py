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

from ..losses.metrics import metric_func

class DynamicModel:
    def __init__(self, name, dataloader, model, optimizer, lp_loss, pde_loss, pde_weight, use_cuda):
        self.name = name
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.lp_loss = lp_loss
        self.pde_loss = pde_loss
        self.pde_weight = pde_weight
        self.use_cuda = use_cuda

        if self.use_cuda:
            model.cuda()
    
    def train(self, epochs, patience, report_interval=500):
        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = np.inf
        best_epoch = 0
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=1e-3, total_steps=epochs, pct_start=0.2)
        for epoch in tqdm(range(epochs)):
            loss_values.append(self.train_step(epoch, report_interval))
            scheduler.step()

            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
                torch.save(self.model.state_dict(), f'out/{self.name}/best_model.pkl')
            else:
                bad_counter += 1

            if bad_counter == patience:
                break

            with open(f'out/{self.name}/outfile.txt', 'a') as fp:
                print(epoch, loss_values[-1], sep='\t', file=fp)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        self.load_model(f'out/{self.name}/best_model.pkl')

        return loss_values

    def report(self, features, targets, loss_train, epoch):
        self.model.eval()
        outputs = self.predict(features, timesteps=targets.shape[-2])

        loss_val = self.get_test_loss(targets, outputs).cpu().item()
        pde_loss_val = self.get_test_pde_loss(features, outputs).cpu().item()
        onestep_loss_val = self.get_test_loss(targets[..., :1, :], outputs[..., :1, :]).cpu().item()
        onestep_pde_loss_val = self.get_test_pde_loss(features, outputs[..., :1, :]).cpu().item()
        with open(f'out/{self.name}/report.txt', 'a') as fp:
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.data.item()),
                'loss_val: {:.4f}'.format(loss_val),
                'pde_loss_val: {:.4f}'.format(pde_loss_val),
                'onestep_loss_val: {:.4f}'.format(onestep_loss_val),
                'onestep_pde_loss_val: {:.4f}'.format(onestep_pde_loss_val),
                'time: {:.4f}'.format(time.time() - self.t_start),
                sep=',', file=fp)

        torch.save(self.model.state_dict(), f'out/{self.name}/model.pkl') 
        self.model.train()
        return loss_val

    def train_step(self, epoch):
        raise NotImplementedError()

    def get_loss(self, features, targets, outputs):
        raise NotImplementedError()

    def predict(self, features, timesteps=1, numpy=False):
        # add with torch.no_grad
        outputs = []
        for i in range(timesteps):
            output = self.model(features)
            features = self.concat_timesteps(features, output)
            if numpy:
                output = output.detach().cpu().numpy()
            outputs.append(output)

        if numpy:
            return np.concatenate(outputs, axis=-2)
        return torch.concat(outputs, dim=-2)

    def get_test_loss(self, targets, outputs):
        total_loss = self.lp_loss(outputs, targets)
        return total_loss

    def get_test_pde_loss(self, features, outputs):
        return self.pde_loss(features, outputs)

    def load_model(self, pkl_file):
        state_dict = torch.load(pkl_file, weights_only=False)
        if '_metadata' in state_dict:
            state_dict.pop('_metadata')
        self.model.load_state_dict(state_dict)

    def concat_timesteps(self, features, outputs):
        t0 = features[..., 1:, :]
        return torch.concat([t0, outputs], dim=-2)
