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

from source.data.dynamic_grid import PDEBenchGrid, PINOGrid
from source.models.onestep_model import OneStepModel
from source.models.pushforward_model import PushForwardModel
from source.models.autoregressive_model import AutoRegressiveModel
from source.models.static_model import StaticModel
from source.losses.navier_stokes_loss import *
from source.losses.normalized_lp_loss import normalized_lp_loss
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
    parser.add_argument('--periodic', action='store_true', default=False, help='Enable periodic boundary conditions.')
    parser.add_argument('--no-train', action='store_true', default=False, help='Disable training (only run prediction).')
    parser.add_argument('--scale-mass', action='store_true', default=False, help='Enable mass scaling for density and pressure.')
    parser.add_argument('--scale-momentum', action='store_true', default=False, help='Enable velocity scaling for momenutm.')
    parser.add_argument('--scale-energy', action='store_true', default=False, help='Enable velocity scaling for kinetic energy.')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for dataloader. If not specified, training will not use batches.')
    parser.add_argument('--model-type', type=str, choices=['OneStep', 'PushForward', 'AutoRegressive'],
                                        default='PushForward', help='Type of training model.')
    parser.add_argument('--pde-weight', type=float, default=1, help='Weighting for the PDE loss.')
    parser.add_argument('--kernel-size', type=int, default=1, help='Kernel size for the convolution part of FNO.')
    parser.add_argument('--nn-arch', type=str, default='FNO', help='Which neural network architecture to use.')
    parser.add_argument('--history', type=int, default=0, help='The number of historical timesteps to use as a history.')
    parser.add_argument('--vorticity', action='store_true', help='Use vorticity equation for incompressible flows.')
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
    if args.vorticity:
        # Load PINO data
        dynamic_data = PINOGrid(args.pdebench, timesteps=args.timesteps,
                                history=args.history, use_cuda=args.cuda)
    else:
        dynamic_data = PDEBenchGrid(args.pdebench, timesteps=args.timesteps,
                                    history=args.history, use_cuda=args.cuda,
                                    reduced_resolution=1, reduced_batch=2)

    np.save(f'GT_train_{dynamic_data.num_channels - 2}D.npy', dynamic_data.train[:100])
    np.save(f'GT_test_{dynamic_data.num_channels - 2}D.npy', dynamic_data.test[:100])

    print(len(dynamic_data))

    if args.cuda:
        dataloader = DataLoader(dynamic_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    else:
        dataloader = DataLoader(dynamic_data, batch_size=args.batch_size, shuffle=True)

    # Model and optimizer
    if args.nn_arch == 'FNO_grad':
        model = PIFNO_grad(ndims=dynamic_data.ndims,
                      nclass=dynamic_data.nfeats,
                      periodic=args.periodic,
                      scale_mass=args.scale_mass,
                      scale_momentum=args.scale_momentum,
                      scale_energy=args.scale_energy,
                      history=args.history,
                      kernel_size=args.kernel_size,
                      static=args.model_type == 'Static')
    else:
        model = PIFNO(ndims=dynamic_data.ndims,
                      nclass=dynamic_data.nfeats,
                      periodic=args.periodic,
                      scale_mass=args.scale_mass,
                      scale_momentum=args.scale_momentum,
                      scale_energy=args.scale_energy,
                      history=args.history,
                      kernel_size=args.kernel_size,
                      static=args.model_type == 'Static')
    
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, 
                        weight_decay=args.weight_decay)

    # Train Model
    def pde_loss(*fargs, **kwargs):
        if args.vorticity:
            loss = pde_iCFD2d_vorticity(*fargs, v=1e-4, **kwargs)
        elif dynamic_data.ndims == 1:
            loss = pde_CFD1d(*fargs, **kwargs)
        elif dynamic_data.ndims == 2:
            loss = pde_CFD2d(*fargs, **kwargs)
        elif dynamic_data.ndims == 3:
            loss = pde_CFD3d(*fargs, **kwargs)
        return loss
    
    def lp_loss(*fargs, **kwargs):
        loss = normalized_lp_loss(*fargs, **kwargs, p=2)
        return loss

    if args.model_type == 'Static':
        dynamic_model = StaticModel(dataloader, model, optimizer, lp_loss, pde_loss, args.pde_weight, args.name, args.cuda)
    elif args.model_type == 'AutoRegressive':
        dynamic_model = AutoRegressiveModel(dataloader, model, optimizer, lp_loss, pde_loss, args.pde_weight, args.name, args.cuda)
    elif args.model_type == 'PushForward':
        dynamic_model = PushForwardModel(dataloader, model, optimizer, lp_loss, pde_loss, args.pde_weight, args.name, args.cuda)
    else: # args.model_type == 'OneStep':
        dynamic_model = OneStepModel(dataloader, model, optimizer, lp_loss, pde_loss, args.pde_weight, args.name, args.cuda)
    
    if args.load_pkl:
        dynamic_model.load_model(args.load_pkl)

    if not args.no_train:
        if os.path.exists(f'out/reports/outfile_{args.name}.txt'):
            os.remove(f'out/reports/outfile_{args.name}.txt')
        if os.path.exists(f'out/reports/report_{args.name}.txt'):
            os.remove(f'out/reports/report_{args.name}.txt')
        train_losses, test_losses = dynamic_model.train(args.epochs, args.patience)

    dynamic_model.model.eval()
    with torch.no_grad():
        if dynamic_data.ndims == 1:
            timesteps = 101 - args.history
        if dynamic_data.ndims == 2:
            timesteps = 21 - args.history
        if dynamic_data.ndims == 3:
            timesteps = 21 - args.history
        
        dynamic_data.timesteps = timesteps

        # dynamic_data.training()
        # locations, features, _ = dynamic_data[0]
        # predictions = dynamic_model.predict(locations, features, timesteps=timesteps, numpy=True)
        # np.save(f'out/predictions/full/{args.name}_full.npy', np.array(predictions))

        dynamic_data.testing()
        locations, features, _ = dynamic_data[0]
        predictions = dynamic_model.predict(locations, features, timesteps=timesteps, numpy=True)
        predictions = np.moveaxis(predictions, -2, 0)
        np.save(f'out/predictions/test/{args.name}_test.npy', np.array(predictions))

        # get train and test metrics

        dynamic_data.testing()
        print(len(dynamic_data))
        loss_info = []
        for i in range(len(dynamic_data)):
            locations, features, targets = dynamic_data[i]

            t0 = time.time()
            predictions = dynamic_model.predict(locations, features, timesteps=timesteps)
            
            test_loss = dynamic_model.get_test_loss(targets, predictions).cpu().item()
            test_pde_loss = dynamic_model.get_test_pde_loss(features, predictions).cpu().item()

            predictions = torch.movedim(predictions, -2, 0)
            targets = torch.movedim(targets, -2, 0)
            t1 = time.time()
            rho_loss = []
            vel_loss = []
            prs_loss = []
            for t, (output, target) in enumerate(zip(predictions, targets)):
                output, target = output.cpu(), target.cpu() 
                rho_loss.append(F.mse_loss(output[..., 0], target[..., 0]).item())
                if not args.vorticity:
                    vel_loss.append(F.mse_loss(output[..., 1:-1], target[..., 1:-1]).item())
                    prs_loss.append(F.mse_loss(output[..., -1], target[..., -1]).item())

            # (batch, nx^i..., timesteps, nc)
            pred = predictions.moveaxis(0, -2).squeeze().unsqueeze(0)
            targets = targets.moveaxis(0, -2).squeeze().unsqueeze(0)

            pred = pred.contiguous().reshape(pred.shape)
            targets.contiguous().reshape(targets.shape)
            if not args.vorticity:
                err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metric_func(pred, targets)

            if args.vorticity:
                loss_info.append({
                    'pred_time': t1 - t0,
                    'test_loss': test_loss,
                    'test_pde_loss': test_pde_loss,
                    'vorticity_loss': rho_loss,
                    })
            else:
                loss_info.append({
                    'pred_time': t1 - t0,
                    'test_loss': test_loss,
                    'test_pde_loss': test_pde_loss,
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

        with open(f'out/predictions/losses/{args.name}_test_losses.json', 'w') as fp:
            json.dump(loss_info, fp)

if __name__ == '__main__':
    main()
