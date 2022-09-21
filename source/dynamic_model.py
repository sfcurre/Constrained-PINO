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
    def __init__(self, dynamic_graph, model, optimizer, ns_loss, use_cuda):
        self.dynamic_graph = dynamic_graph
        self.model = model
        self.optimizer = optimizer
        self.ns_loss = ns_loss
        self.use_cuda = use_cuda

        if self.use_cuda:
            model.cuda()

    def train(self, epochs, patience, timesteps=1):
        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = epochs + 1
        best_epoch = 0
        for epoch in range(epochs):
            loss_values.append(self.train_step(epoch, timesteps=timesteps))

            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
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

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

        # Testing
        test_losses = compute_test(timesteps=timesteps)
        return loss_values, test_losses

    def train_step(self, epoch, timesteps=1):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        for adj, locations, features, targets in tqdm(self.dynamic_graph.sample_train(generator=True, timesteps=timesteps)):
            if use_cuda:
                adj[0].cuda()
                adj[1].cuda()
                locations.cuda()
                features.cuda()
                targets.cuda()

            outputs = self.recursive_predict(locations, features, adj, timesteps=timesteps)

            loss_train = self.recursive_loss(locations, features, targets, outputs)
            loss_train.backward()
            self.optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            outputs = self.recursive_predict(locations, features, adj, timesteps=timesteps)

        loss_val = self.recursive_loss(locations, features, targets, outputs)
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.data.item()),
            'loss_val: {:.4f}'.format(loss_val.data.item()),
            'time: {:.4f}s'.format(time.time() - t))

        return loss_val.data.item()

    def get_loss(self, locations, features, targets, outputs):
        mse_loss = F.mse_loss(outputs, targets)
        ns_loss = self.ns_loss(locations, features, outputs)
        return mse_loss + ns_loss

    def recursive_predict(self, locations, features, adj, timesteps=1):
        outputs = []
        for i in range(timesteps):
            output = self.model(locations, features, adj)
            outputs.append(output)
            features = output
        return outputs

    def recursive_loss(self, locations, features, targets, outputs):
        total_loss = 0
        for output, target in zip(outputs, targets):
            total_loss += self.get_loss(locations, features, target, output)
            features = output
        return total_loss

    def compute_test(self, timesteps=1):
        test_losses = []
        self.model.eval()
        for adj, locations, features, targets in self.dynamic_graph.sample_test(generator=True, timesteps=timesteps):
            outputs = self.recursive_predict(locations, features, adj, timesteps=timesteps)
            loss_test = self.recursive_loss(locations, features, targets, outputs)
            
            test_losses.append(loss_test.data.item())
        return test_losses
