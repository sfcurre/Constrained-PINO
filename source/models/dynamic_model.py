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
    def __init__(self, dataloader, model, optimizer, pde_loss, name, use_cuda):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.pde_loss = pde_loss
        self.name = name
        self.use_cuda = use_cuda

        if self.use_cuda:
            model.cuda()
    
    def train(self, epochs, patience, report_interval=100):
        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = epochs + 1
        best_epoch = 0
        for epoch in tqdm(range(epochs)):
            loss_values.append(self.train_step(epoch, report_interval))

            torch.save(self.model.state_dict(), f'out/saved_models/{self.name}_{epoch}.pkl')
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == patience:
                break

            files = glob.glob(f'out/saved_models/{self.name}_*.pkl')
            for file in files:
                epoch_nb = int(file.split('_')[-1].split('.')[0])
                if epoch_nb < best_epoch:
                    pass
                    #os.remove(file)

            with open(f'outfile_{self.name}.txt', 'a') as fp:
                print(epoch, loss_values[-1], sep='\t', file=fp)

        files = glob.glob(f'out/saved_models/{self.name}_*.pkl')
        for file in files:
            epoch_nb = int(file.split('_')[-1].split('.')[0])
            if epoch_nb > best_epoch:
                pass
                #os.remove(file)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        self.load_model(f'out/saved_models/{self.name}_{best_epoch}.pkl')

        # Testing
        test_losses = self.compute_test()
        return loss_values, test_losses

    def report(self, adj, locations, features, targets, mask, loss_train, epoch):
        self.model.eval()
        outputs = self.predict(locations, features, adj, timesteps=len(targets))

        loss_val = self.get_test_loss(targets, outputs, mask)
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.data.item()),
            'loss_val: {:.4f}'.format(loss_val.data.item()),
            'time: {:.4f}s'.format(time.time() - self.t_start))

        torch.save(self.model.state_dict(), f'out/saved_models/{self.name}_{epoch}.pkl')
        return loss_val.data.item()

    def train_step(self, epoch):
        raise NotImplementedError()

    def get_loss(self, locations, features, targets, outputs, mask):
        raise NotImplementedError()

    def predict(self, locations, features, adj, timesteps=1, numpy=False):
        # add with torch.no_grad
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
        #LOOK AT THIS
        test_losses = []
        self.model.eval()
        for adj, locations, features, targets, _ in self.dataloader:
            outputs = self.predict(locations, features, adj, timesteps=len(targets))
            loss_test = self.get_test_loss(targets, outputs)
            test_losses.append(loss_test.data.item())
        return test_losses

    def get_test_loss(self, targets, outputs, mask = None):
        total_loss = 0
        for output, target in zip(outputs, targets):
            if mask is not None:
                total_loss += torch.sum(((output.cpu() - target.cpu()) * mask.cpu()) ** 2) / torch.sum(mask.cpu())
            else:
                total_loss += F.mse_loss(output.cpu(), target.cpu())
        return total_loss

    def load_model(self, pkl_file):
        self.model.load_state_dict(torch.load(pkl_file))

