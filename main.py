import os
import json
import time
import random
import argparse, yaml
import numpy as np
from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity, record_function

from neuralop.models import FNO
from neuralop.layers.embeddings import GridEmbeddingND 

from source.data.dynamic_grid import PDEBenchGrid, PINOGrid
from source.data.static_grid import PDEBenchStaticGrid, PINOStaticGrid
from source.data.debug_grid import DebugGrid, DebugStaticGrid, DebugNumpyGrid
from source.nn_models import *
from source.custom_fno import CustomFNO
from source.dpot import DPOTNet, AFNO2D
from source.models.onestep_model import OneStepModel
from source.models.pushforward_model import PushForwardModel
from source.models.autoregressive_model import AutoRegressiveModel
from source.models.static_model import StaticModel
from source.models.ar_uncertainty_model import ARUncertaintyModel
from source.models.time_dilated_model import TimeDilatedModel
from source.models.ar_time_dilated_model import ARTimeDilatedModel
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
    parser.add_argument('--eval', action='store_true', default=False, help='Disable training (only run prediction on val set).')
    parser.add_argument('--test', action='store_true', default=False, help='Run predictions on test set.')
    args = parser.parse_args()
    return args

def select_dataset(config, cuda):
    kwargs = dict(timesteps=config['model']['timesteps'],
                  history=config['model']['history'],
                  use_cuda=cuda)
    if config['data']['task'] == 'debug':
        if config['model']['model_type']  in ['Static', 'TimeDilated']:
            data_class = DebugStaticGrid
        else:
            data_class = DebugGrid

    if config['data']['task'] == 'debug_numpy':
        data_class = DebugNumpyGrid
        kwargs['reduced_resolution'] = config['data']['reduce_res']
        
    elif config['data']['task'] == 'pino':
        # Load PINO data
        if config['model']['model_type']  in ['Static', 'TimeDilated']:
            data_class = PINOStaticGrid
        else:
            data_class = PINOGrid
        
    elif config['data']['task'] == 'pdebench':
        # Load PDEBench data
        kwargs['reduced_resolution'] = config['data']['reduce_res']
        kwargs['reduced_batch'] = config['data']['reduce_batch']
        if config['model']['model_type'] == 'Static':
            data_class = PDEBenchStaticGrid
        else:
            data_class = PDEBenchGrid
        
    dynamic_data = data_class(config['data']['data_path'], **kwargs)
    if config['model']['dilation'] > 1 and config['model']['model_type'] in ['TimeDilated', 'ARTimeDilated']:
        dynamic_data.dilate_dt(config['model']['dilation'])
    return dynamic_data

def main(config, args):

    NAME = args.config.split('/')[-1].split('.')[0]
    CUDA = False
    if config['training']['gpus'] > 0 and torch.cuda.is_available():
        print("Using CUDA.")
        CUDA = True
        torch.cuda.manual_seed(1998)

    dynamic_data = select_dataset(config, CUDA)
    
    print('Num batches:', len(dynamic_data))
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

    channels = dynamic_data.nfeats * config['model']['history']
    if config['model']['model_type'] == 'ARTimeDilated' and config['model']['history'] > 1 and config['model']['dilation'] > 1:
        channels = channels * config['model']['dilation'] - dynamic_data.nfeats

    fno = FNO(n_modes=config['model']['modes'],
              in_channels=channels * (1 + dynamic_data.ndims) if config['model']['grad_features'] else channels,
              out_channels=dynamic_data.nfeats,
              hidden_channels=config['model']['hdim'],
              n_layers=config['model']['layers'])
    
    channels_first = True
    if 'dpot' in config['model'] and config['model']['dpot']:
        channels_first = False
        fno = DPOTNet(img_size=dynamic_data.solution.shape[1],
                      patch_size=4,
                      in_channels=dynamic_data.nfeats,
                      out_channels=dynamic_data.nfeats * (1 + dynamic_data.ndims) if config['model']['grad_features'] else dynamic_data.nfeats,
                      in_timesteps=config['model']['history'],
                      out_timesteps=1,
                      embed_dim=config['model']['hdim'],
                      n_blocks=config['model']['layers'],
                      normalize=True)
        
    model = model_class(fno=fno,
                        scale_mass='mass' in config['model']['scaling'],
                        scale_momentum='momentum' in config['model']['scaling'],
                        scale_energy='energy' in config['model']['scaling'],
                        static=config['model']['model_type'] in ['Static', 'TimeDilated'] or not channels_first,
                        uncertainty=config['model']['model_type'] == 'Uncertainty',
                        grad_features=config['model']['grad_features'],
                        channels_first=channels_first)
    
    optimizer = optim.Adam(model.parameters(),
                        lr=config['training']['learning_rate'], 
                        weight_decay=config['training']['weight_decay'])

    # Train Model
    pde_loss = get_pde_loss(dynamic_data.ndims, dynamic_data.spacing, vorticity=config['model']['vorticity'])
    lp_loss = get_lp_loss(dynamic_data.ndims, dynamic_data.variable_weights)
    
    if config['model']['model_type'] == 'Uncertainty':
        model_type = ARUncertaintyModel
        lp_loss = nll_loss
    elif config['model']['model_type'] == 'TimeDilated':
        model_type = partial(TimeDilatedModel, dilation=config['model']['dilation'])
    elif config['model']['model_type'] == 'ARTimeDilated':
        model_type = partial(ARTimeDilatedModel, dilation=config['model']['dilation'])
    elif config['model']['model_type'] == 'Static':
        model_type = StaticModel
    elif config['model']['model_type'] == 'AutoRegressive':
        model_type = AutoRegressiveModel
    elif config['model']['model_type'] == 'PushForward':
        model_type = PushForwardModel
    elif config['model']['model_type'] == 'OneStep':
        model_type = OneStepModel

    dynamic_model = model_type(NAME, dataloader, model, optimizer, lp_loss, pde_loss, config['model']['pde_weight'], CUDA)
    
    if args.load:
        dynamic_model.load_model(args.load)

    if not os.path.exists(f'out/{NAME}'):
        os.makedirs(f'out/{NAME}')

    if not args.eval:
        if os.path.exists(f'out/{NAME}/outfile.txt'):
            os.remove(f'out/{NAME}/outfile.txt')
        if os.path.exists(f'out/{NAME}/report.txt'):
            os.remove(f'out/{NAME}/report.txt')

        train_losses = dynamic_model.train(config['training']['epochs'], config['training']['patience'])            


    dynamic_model.model.eval()
    with torch.no_grad():
        timesteps = dynamic_data.duration - config['model']['history']
        dynamic_data.timesteps = timesteps

        dynamic_data.testing(test=args.test)
        preds, targs = [], []
        for i in range(min(len(dynamic_data), 10)):
            features, targets = dynamic_data[i]
            print(features.shape)
            predictions = dynamic_model.predict(features.unsqueeze(0), timesteps=timesteps, numpy=True)[0]
            predictions = np.moveaxis(predictions, -2, 0)
            targets = np.moveaxis(targets.detach().cpu().numpy(), -2, 0)
            initial = np.moveaxis(features[..., -1:, -predictions.shape[-1]:].detach().cpu().numpy(), -2, 0)
            preds.append(np.concat([initial, predictions], axis=0))
            targs.append(np.concat([initial, targets], axis=0))

        predictions = np.stack(preds, axis=0)
        targets = np.stack(targs, axis=0)

        np.save(f'out/{NAME}/test_preds.npy', predictions)
        np.save(f'out/{NAME}/test_targets.npy', targets)

        # get train and test metrics

        dynamic_data.testing(test=args.test)
        loss_info = []
        for i in range(len(dynamic_data)):
            features, targets = dynamic_data[i]
            features = features.unsqueeze(0)
            targets = targets.unsqueeze(0)

            t0 = time.time()
            predictions = dynamic_model.predict(features, timesteps=timesteps)
            
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
                if not config['model']['vorticity']:
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

        with open(f'out/{NAME}/test_losses.json', 'w') as fp:
            json.dump(loss_info, fp)

if __name__ == '__main__':
    args = parse_args()
    config_file = args.config
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    main(config, args)
