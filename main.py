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
# from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter()

from source.dynamic_graph import DynamicGraph3D
from source.dynamic_model import DynamicModel
from source.losses.navier_stokes_loss import NavierStokesLoss3D
from source.models import PeriodicSpGAT, PeriodicSpGATTorchGeo

# Example call:
# python main.py --grd TGV/ijk_snapshots/ijkmvmv_grd.npy --sol TGV/ijk_snapshots/ijkmvmv_sol.npy

# Code collected from https://github.com/Diego999/pyGAT

# Training settings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grd', type=str, help='Path to grd data file.')
    parser.add_argument('--sol', type=str, help='Beginning of solution filename(s).')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb-heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--timesteps', type=int, default=5, help='Number of timesteps to train over.')
    parser.add_argument('--load-pkl', type=str, default='', help='Existing model state to load initially.')
    parser.add_argument('--no-train', action='store_true', default=False, help='Disable training (only run prediction).')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def main():

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        print("Using CUDA.")
        torch.cuda.manual_seed(args.seed)

    # Load data
    dynamic_data = DynamicGraph3D(args.grd, args.sol, periodic=True, use_cuda=args.cuda)

    # Model and optimizer
    model = PeriodicSpGATTorchGeo(nfeat=5,
                nhid=args.hidden, 
                nclass=5, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, 
                        weight_decay=args.weight_decay)

    # Train Model
    ns_loss = NavierStokesLoss3D(10, 1)
    dynamic_model = DynamicModel(dynamic_data, model, optimizer, ns_loss, args.cuda)

    if args.load_pkl:
        dynamic_model.load_model(args.load_pkl)

    if not args.no_train:
        train_losses, test_losses = dynamic_model.train(args.epochs, args.patience, timesteps=args.timesteps)

    dynamic_model.model.eval()
    with torch.no_grad():
        adj, locations, features, _ = dynamic_data.sample_train(t=0)
        predictions = dynamic_model.recursive_predict(locations, features, adj, timesteps=1000, evaluate=True, numpy=True)
        np.save('full_predictions.npy', np.array(predictions))

        adj, locations, features, _ = dynamic_data.sample_test(t=0)
        predictions = dynamic_model.recursive_predict(locations, features, adj, timesteps=200, evaluate=True, numpy=True)
        np.save('test_predictions.npy', np.array(predictions))

if __name__ == '__main__':
    main()
