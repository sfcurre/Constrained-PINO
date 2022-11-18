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

class DynamicModel:
    def __init__(self, dataloader, model, optimizer, pde_loss, use_cuda):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.pde_loss = pde_loss
        self.use_cuda = use_cuda

        if self.use_cuda:
            model.cuda()
    
    def train(self, epochs, patience):
        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = epochs + 1
        best_epoch = 0
        for epoch in tqdm(range(epochs)):
            loss_values.append(self.train_step(epoch))

            torch.save(self.model.state_dict(), '{}.pkl'.format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == patience:
                break

            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)

            with open('outfile.txt', 'a') as fp:
                print(epoch, loss_values[-1], sep='\t', file=fp)

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        self.load_model('{}.pkl'.format(best_epoch))

        # Testing
        test_losses = self.compute_test()
        return loss_values, test_losses

    def train_step(self, epoch):
        raise NotImplementedError()

    def get_loss(self, locations, features, targets, outputs):
        raise NotImplementedError()

    def predict(self, locations, features, adj, timesteps=1, numpy=False):
        outputs = []
        for i in range(timesteps):
            output = self.model(locations, features, adj)
            features = output
            output = output.detach().cpu()
            if numpy:
                output = output.numpy()
            outputs.append(output)
        return outputs

    def compute_test(self):
        test_losses = []
        self.model.eval()
        for adj, locations, features, targets in self.dataloader:
            outputs = self.predict(locations, features, adj, timesteps=len(targets))
            loss_test = self.get_test_loss(targets, outputs)
            test_losses.append(loss_test.data.item())
        return test_losses

    def get_test_loss(self, targets, outputs):
        total_loss = 0
        for output, target in zip(outputs, targets):
            total_loss += F.mse_loss(output.cpu(), target.cpu())
        return total_loss

    def load_model(self, pkl_file):
        self.model.load_state_dict(torch.load(pkl_file))

