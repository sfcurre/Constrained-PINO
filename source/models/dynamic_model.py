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
    def __init__(self, dataloader, model, optimizer, pde_loss, pde_weight, name, use_cuda):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.pde_loss = pde_loss
        self.pde_weight = pde_weight
        self.name = name
        self.use_cuda = use_cuda

        if self.use_cuda:
            model.cuda()
    
    def train(self, epochs, patience, report_interval=1000):
        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = epochs + 1
        best_epoch = 0
        for epoch in tqdm(range(epochs)):
            loss_values.append(self.train_step(epoch, report_interval))

            #torch.save(self.model.state_dict(), f'out/saved_models/{self.name}_{epoch}.pkl')
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
                torch.save(self.model.state_dict(), f'out/saved_models/{self.name}.pkl')
            else:
                bad_counter += 1

            if bad_counter == patience:
                break

            with open(f'out/reports/outfile_{self.name}.txt', 'a') as fp:
                print(epoch, loss_values[-1], sep='\t', file=fp)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        self.load_model(f'out/saved_models/{self.name}.pkl')

        # Testing
        test_losses = self.compute_test()
        return loss_values, test_losses

    def report(self, locations, features, targets, loss_train, epoch):
        self.model.eval()
        outputs = self.predict(locations, features, timesteps=len(targets))

        # pred = torch.stack(outputs, dim=-2).squeeze().unsqueeze(0)
        # metric_targets = targets.moveaxis(0, -2).squeeze().unsqueeze(0)

        # pred = pred.contiguous().reshape(pred.shape)
        # err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metric_func(pred, metric_targets)

        loss_val = self.get_test_loss(targets, outputs)
        pde_loss_val = self.get_test_pde_loss(locations, features, outputs)
        with open(f'out/reports/report_{self.name}.txt', 'a') as fp:
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.data.item()),
                'loss_val: {:.4f}'.format(loss_val.data.item()),
                'pde_loss_val: {:.4f}'.format(pde_loss_val.data.item()),
                'time: {:.4f}'.format(time.time() - self.t_start),
                # 'RMSE: {:.4f}'.format(err_RMSE.cpu().item()),
                # 'nRMSE: {:.4f}'.format(err_nRMSE.cpu().item()),
                # 'err_CSV: {:.4f}'.format(err_CSV.cpu().item()),
                # 'err_Max: {:.4f}'.format(err_Max.cpu().item()),
                # 'err_BD: {:.4f}'.format(err_BD.cpu().item()),
                # 'err_F: {:.4f}'.format(err_F.cpu().mean().item()),
                sep=',', file=fp)

        torch.save(self.model.state_dict(), f'out/saved_models/{self.name}_{epoch}.pkl')
        return loss_val.data.item()

    def train_step(self, epoch):
        raise NotImplementedError()

    def get_loss(self, locations, features, targets, outputs):
        raise NotImplementedError()

    def predict(self, locations, features, timesteps=1, numpy=False):
        # add with torch.no_grad
        outputs = []
        for i in range(timesteps):
            output = self.model(locations, features)
            features = self.concat_timesteps(features, output)
            output = output.detach().cpu()
            if numpy:
                output = output.numpy()
            outputs.append(output)
        return outputs

    def compute_test(self):
        #LOOK AT THIS
        test_losses = []
        self.model.eval()
        for adj, locations, features, targets, _ in self.dataloader:
            outputs = self.predict(locations, features, adj, timesteps=len(targets))
            loss_test = self.get_test_loss(targets, outputs)
            test_losses.append(loss_test.data.item())
        return test_losses

    def get_test_loss(self, targets, outputs):
        total_loss = 0
        for output, target in zip(outputs, targets):
            total_loss += F.mse_loss(output.cpu(), target.cpu())
        return total_loss

    def get_test_pde_loss(self, locations, features, outputs):
        total_loss = 0
        for i in range(outputs):
            if i == 0:
                total_loss += self.pde_loss(locations.cpu(), features.cpu(), outputs[i].cpu())
            else:
                total_loss += self.pde_loss(locations.cpu(), outputs[i-1].cpu(), outputs[i].cpu())
        return total_loss

    def load_model(self, pkl_file):
        self.model.load_state_dict(torch.load(pkl_file))

    def concat_timesteps(self, features, outputs):
        t0 = features[..., outputs.shape[-1]:]
        return torch.concat([t0, outputs], dim=-1)
