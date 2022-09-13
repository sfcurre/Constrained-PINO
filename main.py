import os
import glob
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from source.dynamic_graph import DynamicGraph3D
from source.models import GAT, SpGAT

# Code collected from https://github.com/Diego999/pyGAT

# Training settings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grd', type=str, help='Path to grd data file.')
    parser.add_argument('--sol', type=str, help='Beginning of solution filename(s).')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def main():

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    dynamic_data = DynamicGraph3D(args.grd, args.sol, periodic=False)

    grid = dynamic_data.grid
    print(grid.shape)

    locations = grid.reshape((-1, grid.shape[-1]))
    print(locations.shape)

    trig_locations = dynamic_data.locations
    print(locations.shape)

    trig_grid = trig_locations.reshape((grid.shape[0], grid.shape[1], grid.shape[2], locations.shape[-1]))
    print(trig_grid.shape)

    t0 = dynamic_data.sample_train(t=0)
    print(t0.shape)

    fig = plt.figure()
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=locations[:, i], alpha=0.5)
    plt.show()

    # fig = plt.figure()
    # for i in range(6):
    #     ax = fig.add_subplot(1, 6, i + 1, projection='3d')
    #     ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=trig_locations[:, i], alpha=0.5)
    # plt.show()

    fig = plt.figure()
    for i in range(5):
        ax = fig.add_subplot(1, 5, i + 1, projection='3d')
        ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=t0[:, 3 + i], alpha=0.5)
    plt.show()


    return
    # Model and optimizer
    if args.sparse:
        model = SpGAT(nfeat=features.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(labels.max()) + 1, 
                    dropout=args.dropout, 
                    nheads=args.nb_heads, 
                    alpha=args.alpha)
    else:
        model = GAT(nfeat=features.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(labels.max()) + 1, 
                    dropout=args.dropout, 
                    nheads=args.nb_heads, 
                    alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), 
                        lr=args.lr, 
                        weight_decay=args.weight_decay)

if __name__ == '__main__':
    main()