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
from torch.utils.data import DataLoader
# from torch.profiler import profile, record_function, ProfilerActivity
# from pytorch_memlab import LineProfiler

# from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter()

from source.data.dynamic_graph import DynamicGraph
from source.data.periodic_graph import PeriodicGraph
from source.data.tgv_graph import TGVGraph, PeriodicTGVGraph, MiniBatchTGVGraph
from source.data.pdebench_graph import PDEBenchGraph, PeriodicPDEBenchGraph, MiniBatchPDEBenchGraph
from source.models.onestep_model import OneStepModel
from source.models.pushforward_model import PushForwardModel
from source.models.autoregressive_model import AutoRegressiveModel
from source.losses.navier_stokes_loss import pde_CFD3d, pde_CFD2d
from source.nn_models import DeepSpGAT, PIGNN

# Example call:
# python main.py --grd TGV/ijk_snapshots/ijkmvmv_grd.npy --sol TGV/ijk_snapshots/ijkmvmv_sol.npy --name ExampleMain --model-type OneStep

# torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Model name for saving files.')
    parser.add_argument('--grd', type=str, help='Path to grd data file.')
    parser.add_argument('--sol', type=str, help='Beginning of solution filename(s).')
    parser.add_argument('--pdebench', type=str, help='Filename for pdebench dataset.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers in the neural network.')
    parser.add_argument('--nb-heads', type=int, default=4, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--timesteps', type=int, default=5, help='Number of timesteps to train over.')
    parser.add_argument('--load-pkl', type=str, default='', help='Existing model state to load initially.')
    parser.add_argument('--no-train', action='store_true', default=False, help='Disable training (only run prediction).')
    parser.add_argument('--periodic-bc', action='store_true', default=False, help='Enable a periodic boundary condition.')
    parser.add_argument('--mini-batch-width', type=int, default=0, help='Enable minibatch training with input width (requires periodic BC)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for dataloader. If not specified, training will not use batches.')
    parser.add_argument('--model-type', type=str, choices=['OneStep', 'PushForward', 'AutoRegressive'],
                                        default='PushForward', help='Type of training model.')
    parser.add_argument('--pde-weight', type=float, default=1, help='Weighting for the PDE loss.')
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
    if args.grd and args.sol:
        if args.periodic_bc:
            if args.mini_batch_width:
                dynamic_data = MiniBatchTGVGraph(args.grd, args.sol, args.mini_batch_width,
                                                 args.layers, args.timesteps, use_cuda=args.cuda)
            else:
                dynamic_data = PeriodicTGVGraph(args.grd, args.sol, args.timesteps, use_cuda=args.cuda)
        else:
            dynamic_data = TGVGraph(args.grd, args.sol, args.timesteps, use_cuda=args.cuda)
    elif args.pdebench:
        if args.periodic_bc:
            if args.mini_batch_width:
                dynamic_data = MiniBatchPDEBenchGraph(args.pdebench, args.mini_batch_width,
                                                      args.layers, args.timesteps, use_cuda=args.cuda)
            else:
                dynamic_data = PeriodicPDEBenchGraph(args.pdebench, timesteps=args.timesteps, use_cuda=args.cuda,
                                                     reduced_resolution=4, reduced_batch=2)
        else:
            dynamic_data = PDEBenchGraph(args.pdebench, timesteps=args.timesteps, use_cuda=args.cuda)

    if args.cuda:
        dataloader = DataLoader(dynamic_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    else:
        dataloader = DataLoader(dynamic_data, batch_size=args.batch_size, shuffle=True)

    # Model and optimizer
    attn_model = DeepSpGAT(nfeat=dynamic_data.nfeats,
                ndims=dynamic_data.ndims,
                nhid=args.hidden,
                natt=args.layers,
                nclass=dynamic_data.nfeats,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha,
                periodic=args.periodic_bc)
    gcn_model = PIGNN(nfeat=dynamic_data.nfeats,
                ndims=dynamic_data.ndims,
                nhid=args.hidden,
                nlayers=args.layers,
                nclass=dynamic_data.nfeats,
                dropout=args.dropout,
                periodic=args.periodic_bc)
    #model = attn_model
    #del gcn_model
    model = gcn_model
    del attn_model
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, 
                        weight_decay=args.weight_decay)

    # Train Model
    def pde_loss(*fargs, **kwargs):
        loss = pde_CFD2d(*fargs, **kwargs)
        return loss * args.pde_weight

    if args.model_type == 'OneStep':
        dynamic_model = OneStepModel(dataloader, model, optimizer, pde_loss, args.name, args.cuda)
    elif args.model_type == 'AutoRegressive':
        dynamic_model = AutoRegressiveModel(dataloader, model, optimizer, pde_loss, args.name, args.cuda)
    else:
        dynamic_model = PushForwardModel(dataloader, model, optimizer, pde_loss, args.name, args.cuda)

    if args.load_pkl:
        dynamic_model.load_model(args.load_pkl)

    if not args.no_train:
        train_losses, test_losses = dynamic_model.train(args.epochs, args.patience)

    dynamic_model.model.eval()
    with torch.no_grad():
        dynamic_data.training()
        adj, locations, features, _, _ = dynamic_data[0]
        predictions = dynamic_model.predict(locations, features, adj, timesteps=30, numpy=True)
        np.save(f'out/predictions/{args.name}_full.npy', np.array(predictions))

        dynamic_data.testing()
        adj, locations, features, _, _ = dynamic_data[0]
        predictions = dynamic_model.predict(locations, features, adj, timesteps=30, numpy=True)
        np.save(f'out/predictions/{args.name}_test.npy', np.array(predictions))

if __name__ == '__main__':
    main()
