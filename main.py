import os
import json
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

from source.data.dynamic_grid import PDEBenchGrid, TGVGrid
from source.models.onestep_model import OneStepModel
from source.models.pushforward_model import PushForwardModel
from source.models.autoregressive_model import AutoRegressiveModel
from source.losses.navier_stokes_loss import *
from source.nn_models import *

from source.losses.metrics import metric_func

# Example call:
# python main.py --grd TGV/ijk_snapshots/ijkmvmv_grd.npy --sol TGV/ijk_snapshots/ijkmvmv_sol.npy --name ExampleMain --model-type OneStep

# torch.autograd.set_detect_anomaly(True)

# np.random.seed(1998)
# torch.manual_seed(1998)

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
    parser.add_argument('--scale-totals', action='store_true', default=False, help='Enable total scaling for density and pressure.')
    parser.add_argument('--mini-batch-width', type=int, default=0, help='Enable minibatch training with input width (requires periodic BC)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for dataloader. If not specified, training will not use batches.')
    parser.add_argument('--model-type', type=str, choices=['OneStep', 'PushForward', 'AutoRegressive'],
                                        default='PushForward', help='Type of training model.')
    parser.add_argument('--pde-weight', type=float, default=1, help='Weighting for the PDE loss.')
    parser.add_argument('--use-grid', action='store_true', help='Flag indicating if the program should use grids instead of graphs.')
    parser.add_argument('--nn-arch', type=str, default='PIGNN', help='Which neural network architecture to use.')
    parser.add_argument('--history', type=int, default=0, help='The number of historical timesteps to use as a history.')
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
    dynamic_data = PDEBenchGrid(args.pdebench, timesteps=args.timesteps,
                                history=args.history, use_cuda=args.cuda,
                                reduced_resolution=2, reduced_batch=2)

    if args.cuda:
        dataloader = DataLoader(dynamic_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    else:
        dataloader = DataLoader(dynamic_data, batch_size=args.batch_size, shuffle=True)

    # Model and optimizer
    if args.nn_arch == 'DeepSpGAT':
        model = DeepSpGAT(nfeat=dynamic_data.nfeats,
                    ndims=dynamic_data.ndims,
                    nhid=args.hidden,
                    natt=args.layers,
                    nclass=dynamic_data.nfeats,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha,
                    periodic=args.periodic_bc)
    elif args.nn_arch == 'FNO':
        model = PIFNO(ndims=dynamic_data.ndims,
                      nclass=dynamic_data.nfeats,
                      periodic=args.periodic_bc,
                      scale=args.scale_totals,
                      history=args.history)
    else:
        # args.nn_arch == 'PIGNN'
        model = PIGNN(nfeat=dynamic_data.nfeats,
                    ndims=dynamic_data.ndims,
                    nhid=args.hidden,
                    nlayers=args.layers,
                    nclass=dynamic_data.nfeats,
                    dropout=args.dropout,
                    periodic=args.periodic_bc)
    
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, 
                        weight_decay=args.weight_decay)

    # Train Model
    def pde_loss(*fargs, **kwargs):
        if dynamic_data.ndims == 1:
            loss = pde_CFD1d_approx(*fargs, **kwargs)
        if dynamic_data.ndims == 2:
            loss = pde_CFD2d_approx(*fargs, **kwargs)
        if dynamic_data.ndims == 3:
            loss = pde_CFD3d_approx(*fargs, **kwargs)
        return loss

    if args.model_type == 'OneStep':
        dynamic_model = OneStepModel(dataloader, model, optimizer, pde_loss, args.pde_weight, args.name, args.cuda)
    elif args.model_type == 'AutoRegressive':
        dynamic_model = AutoRegressiveModel(dataloader, model, optimizer, pde_loss, args.pde_weight, args.name, args.cuda)
    else:
        dynamic_model = PushForwardModel(dataloader, model, optimizer, pde_loss, args.pde_weight, args.name, args.cuda)

    if args.load_pkl:
        dynamic_model.load_model(args.load_pkl)

    if not args.no_train:
        train_losses, test_losses = dynamic_model.train(args.epochs, args.patience)

    dynamic_model.model.eval()
    with torch.no_grad():
        if dynamic_data.ndims == 1:
            timesteps = 91
        if dynamic_data.ndims == 2:
            timesteps = 11
        if dynamic_data.ndims == 3:
            timesteps = 11
        
        dynamic_data.timesteps = timesteps

        dynamic_data.training()
        adj, locations, features, _, _ = dynamic_data[0]
        predictions = dynamic_model.predict(locations, features, adj, timesteps=timesteps, numpy=True)
        np.save(f'out/predictions/{args.name}_full.npy', np.array(predictions))

        dynamic_data.testing()
        adj, locations, features, _, _ = dynamic_data[0]
        predictions = dynamic_model.predict(locations, features, adj, timesteps=timesteps, numpy=True)
        np.save(f'out/predictions/{args.name}_test.npy', np.array(predictions))

        # get train and test metrics

        dynamic_data.testing()
        print(len(dynamic_data))
        loss_info = []
        for i in range(len(dynamic_data)):
            adj, locations, features, targets, _ = dynamic_data[i]

            t0 = time.time()
            predictions = dynamic_model.predict(locations, features, adj, timesteps=timesteps)
            t1 = time.time()
            rho_loss = []
            vel_loss = []
            prs_loss = []
            for t, (output, target) in enumerate(zip(predictions, targets)):
                output, target = output.cpu(), target.cpu() 
                rho_loss.append(F.mse_loss(output[..., 0], target[..., 0]).item())
                vel_loss.append(F.mse_loss(output[..., 1:-1], target[..., 1:-1]).item())
                prs_loss.append(F.mse_loss(output[..., -1], target[..., -1]).item())

            # (batch, nx^i..., timesteps, nc)
            pred = torch.stack(predictions, dim=-2).squeeze().unsqueeze(0)
            targets = targets.moveaxis(0, -2).squeeze().unsqueeze(0)

            pred = pred.contiguous().reshape(pred.shape)
            targets.contiguous().reshape(targets.shape)
            err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metric_func(pred, targets)

            loss_info.append({
                'pred_time': t1 - t0,
                'density_loss': rho_loss,
                'velocity_loss': vel_loss,
                'pressure_loss': prs_loss,
                'RMSE': err_RMSE.cpu().item(),
                'nRMSE': err_nRMSE.cpu().item(),
                'err_CSV': err_CSV.cpu().item(),
                'err_Max': err_Max.cpu().item(),
                'err_BD': err_BD.cpu().item(),
                'err_F': err_F.cpu().mean().item()
            })

        with open(f'out/predictions/{args.name}_test_losses.json', 'w') as fp:
            json.dump(loss_info, fp)

if __name__ == '__main__':
    main()
