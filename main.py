import os
import json
import time
import random
import argparse, yaml
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from source.data.dynamic_grid import PDEBenchGrid, PINOGrid
from source.data.static_grid import PDEBenchStaticGrid, PINOStaticGrid
from source.nn_models import *
from source.models.onestep_model import OneStepModel
from source.models.pushforward_model import PushForwardModel
from source.models.autoregressive_model import AutoRegressiveModel
from source.models.static_model import StaticModel
from source.models.ar_uncertainty_model import ARUncertaintyModel
from source.models.time_dilated_model import TimeDilatedModel
from source.losses.navier_stokes_loss import *
from source.losses.lp_loss import *
from source.losses.metrics import metric_func

random.seed(1998)
np.random.seed(1998)
torch.manual_seed(1998)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to config file.')
    parser.add_argument('--load', type=str, default='', help='Existing model state to load initially.')
    parser.add_argument('--eval', action='store_true', default=False, help='Disable training (only run prediction).')
    args = parser.parse_args()
    return args

def select_dataset(config, cuda):
    kwargs = dict(timesteps=config['model']['timesteps'],
                  use_cuda=cuda)
    if config['data']['task'] == 'pino':
        # Load PINO data
        if config['model']['model_type'] == 'Static':
            data_class = PINOStaticGrid
        else:
            data_class = PINOGrid
            kwargs['history'] = config['model']['history']
    
    elif config['data']['task'] == 'pdebench':
        if args.model_type == 'Static':
            data_class = PDEBenchStaticGrid
            kwargs['reduced_resolution'] = config['data']['reduce_res']
            kwargs['reduce_batch'] = config['data']['reduced_batch']   
        else:
            data_class = PDEBenchGrid
            kwargs['history'] = config['model']['history']         
            kwargs['reduced_resolution'] = config['data']['reduce_res']
            kwargs['reduce_batch'] = config['data']['reduced_batch']
    dynamic_data = data_class(config['data']['data_path'], **kwargs)
    return dynamic_data

def main(config, args):

    CUDA = False
    if config['training']['gpus'] > 0 and torch.cuda.is_available():
        print("Using CUDA.")
        CUDA = True
        torch.cuda.manual_seed(1998)

    dynamic_data = select_dataset(config, CUDA)
    
    np.save(f'GT_train_{dynamic_data.ndims}D{"_vorticity" * args.vorticity}.npy', dynamic_data.train[:100])
    np.save(f'GT_test_{dynamic_data.ndims}D{"_vorticity" * args.vorticity}.npy', dynamic_data.test[:100])

    print('Num batches:', len(dynamic_data))

    if CUDA:
        dataloader = DataLoader(dynamic_data, batch_size=config['training']['batch_size'], shuffle=True, generator=torch.Generator(device='cuda'))
    else:
        dataloader = DataLoader(dynamic_data, batch_size=config['training']['batch_size'], shuffle=True)

    # Model and optimizer
    if config['model']['vorticity']:
        if config['model']['grad']:
            model_class = iPIFNO_grad
        else:
            model_class = iPIFNO
    else:
        if config['model']['grad']:
            model_class = PIFNO_grad
        else:
            model_class = PIFNO    

    model = model_class(ndims=dynamic_data.ndims,
                        nclass=dynamic_data.nfeats,
                        periodic=config['model']['periodic'],
                        scale_mass='mass' in config['model']['scaling'],
                        scale_momentum='momentum' in config['model']['scaling'],
                        scale_energy='energy' in config['model']['scaling'],
                        history=config['model']['history'],
                        kernel_size=config['model']['kernel_size'],
                        static=config['model']['model_type'] == 'Static',
                        uncertainty=config['model']['model_type'] == 'Uncertainty')
    
    optimizer = optim.Adam(model.parameters(),
                        lr=config['training']['learning_rate'], 
                        weight_decay=config['training']['weight_decay'])

    # Train Model
    pde_loss = get_pde_loss(dynamic_data.ndims, vorticity=config['model']['vorticity'])
    lp_loss = get_lp_loss(dynamic_data.ndims)
    
    if args.model_type == 'Uncertainty':
        model_type = ARUncertaintyModel
        lp_loss = nll_loss
    elif args.model_type == 'TimeDilated':
        model_type = partial(TimeDilatedModel, dilation=config['model']['dilation'])
    elif args.model_type == 'Static':
        model_type = StaticModel
    elif args.model_type == 'AutoRegressive':
        model_type = AutoRegressiveModel
    elif args.model_type == 'PushForward':
        model_type = PushForwardModel
    elif args.model_type == 'OneStep':
        model_type = OneStepModel

    dynamic_model = model_type(args.name, dataloader, model, optimizer, lp_loss, pde_loss, config['model']['pde_weight'], CUDA)
    
    if args.load:
        dynamic_model.load_model(args.load)

    if not args.eval:
        if os.path.exists(f'out/reports/outfile_{args.name}.txt'):
            os.remove(f'out/reports/outfile_{args.name}.txt')
        if os.path.exists(f'out/reports/report_{args.name}.txt'):
            os.remove(f'out/reports/report_{args.name}.txt')
        train_losses, test_losses = dynamic_model.train(config['training']['epochs'], config['training']['patience'])

    dynamic_model.model.eval()
    with torch.no_grad():
        timesteps = dynamic_data.duration - config['model']['history']
        dynamic_data.timesteps = timesteps

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
            if not config['model']['vorticity']:
                err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = metric_func(pred, targets)

            if config['model']['vorticity']:
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
    args = parse_args()
    config_file = args.config
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    main(config, args)
