import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import os

from collections import defaultdict

sns.set_theme()

def read_loss_file(fname):
    with open(fname) as fp:
        data = json.load(fp)

    losses = defaultdict(list)
    for sample in data:
        for term in sample:
            losses[term].append(sample[term])

    for term in losses:
        arr = np.array(losses[term])
        losses[term] = arr.mean(axis=0), arr.std(axis=0) 
    
    return losses

def organize_info(fnames, terms):
    all_info = {}
    for fname in fnames:
        losses = read_loss_file(fname)
        dataset, model, lr, lr2, pde_weight, scaling = fname.split('-')
        if lr2.endswith('6'):
            continue

        for term in terms:
            if term not in all_info:
                all_info[term] = defaultdict(list)
            
            if type(term) is str:
                mean, std = losses[term]
            elif type(term) is tuple:
                mean, std = 0, None
                for t in term:
                    mean += losses[t][0]

            info = {"mean": mean, "std": std, "model": model}
            

            if model == 'OneStep':
                label = ''
                colors = sns.color_palette()[1::2]
            else:
                label = 'AR '
                colors = sns.color_palette()[0::2]

            if pde_weight.endswith('0.0'):
                label += 'FNO '
                color = colors[0]
                
            elif pde_weight.endswith('0.01'):
                label += 'PI-FNO '
                color = colors[1]
                
            if scaling.startswith('un'):
                label += '(unscaled)'
                linestyle = '--'
            elif 'ke' in scaling:
                label += '(KE scaled)'
                linestyle = ':'
            else:
                label += '(scaled)'
                linestyle = '-'
    
            info['color'] = color
            info['linestyle'] = linestyle
            info['label'] = label

            all_info[term][dataset.split('_')[1]].append(info)

    return all_info

def construct_table(fnames, dimension='3D'):
    metrics = ['RMSE', 'nRMSE', 'err_Max', 'err_CSV', 'err_BD', 'err_F']
    metric_grid = []
    indices = []
    ke = False
    for fname in fnames:
        dataset, model, lr, lr2, pde_weight, scaling = fname.split('-')
        if dimension not in dataset or lr2.endswith('6'):
            continue
        losses = read_loss_file(fname)
        
        f_metrics = []
        for metric in metrics:
            mean, std = losses[metric]
            met = f'{mean:.3f} $\\pm$ {std:.3f}'
            f_metrics.append(met)

        if pde_weight.endswith('0.0'):
            label = 'FNO '
        elif pde_weight.endswith('0.01'):
            label = 'PI-FNO '
        
        if scaling.startswith('un'):
            label += '(unscaled)'
        elif 'ke' in scaling:
                continue
                label += '(KE scaled)'
                ke = True
        else:
            label += '(scaled)'

        index = (model, label)
        metric_grid.append(f_metrics)
        indices.append(index) 

    index = pd.MultiIndex.from_tuples(indices, names=['Training', 'Model'])
    df = pd.DataFrame(metric_grid, index=index, columns=metrics)
    df = df.transpose()

    # reorder df
    upper = ['OneStep', 'AutoRegressive']
    lower = ['FNO (unscaled)', 'FNO (scaled)', 'PI-FNO (unscaled)', 'PI-FNO (scaled)']
    if ke:
        lower = ['FNO (unscaled)', 'FNO (scaled)', 'FNO (KE scaled)', 'PI-FNO (unscaled)', 'PI-FNO (scaled)', 'PI-FNO (KE scaled)']
    columns = [(u, l) for u in upper for l in lower]
    columns = pd.MultiIndex.from_tuples(columns, names=['Training', 'Model'])
    df = pd.DataFrame(df, columns=columns)
    df.index = ['RMSE', 'nRMSE', 'MaxError', 'cRMSE', 'bRMSE', 'fRMSE']
    return df

#=========================================================
def main():
    
    file_dir = 'out/predictionsFullScale/'
    filenames = os.listdir(file_dir)
    filenames = [file_dir + f for f in filenames]

    terms = ['density_loss', 'velocity_loss', 'pressure_loss']
    terms.append(tuple(terms))
    
    all_info = organize_info(filenames, terms)

    limits = {'1D': 0.8, '2D': 0.8, '3D': 0.3}
    for dimension in ['2D']: #['1D', '2D', '3D']:
        for term in terms:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            for info in all_info[term][dimension]:
                ax = axes[int(info['model'] == 'AutoRegressive')]
                ax.plot(np.arange(len(info['mean'])),
                        info['mean'],
                        color=info['color'],
                        linestyle=info['linestyle'],
                        label=info['label'])
                
            
            axes[0].set_xlabel('Time', fontsize=16)
            axes[0].set_ylabel('MSE Loss', fontsize=16)
            axes[1].set_xlabel('Time', fontsize=16)
            axes[1].set_ylabel('MSE Loss', fontsize=16)

            axes[0].set_xlim(0, len(info['mean']) - 1)
            # axes[0].set_ylim(0, limits[dimension])
            axes[1].set_xlim(0, len(info['mean']) - 1)
            # axes[1].set_ylim(0, limits[dimension])

            axes[0].legend(loc='upper left')
            axes[1].legend(loc='upper left')
            
            axes[0].set_title('OneStep Models', fontsize=20)
            axes[1].set_title('AutoRegressive Models', fontsize=20)

            if type(term) is tuple:
                term = 'combined'
            plt.savefig(f'visuals/AR_losses/{dimension}_{term}.png', bbox_inches='tight')
            del fig

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for i, term in enumerate([('density_loss', 'velocity_loss', 'pressure_loss'), 'velocity_loss']):
            for info in all_info[term][dimension]:
                ax = axes[i]
                ax.plot(np.arange(len(info['mean'])),
                        info['mean'],
                        color=info['color'],
                        linestyle=info['linestyle'],
                        label=info['label'])
            
        axes[0].set_xlabel('Time', fontsize=16)
        axes[0].set_ylabel('MSE Loss', fontsize=16)
        axes[1].set_xlabel('Time', fontsize=16)
        axes[1].set_ylabel('MSE Loss', fontsize=16)

        axes[0].set_xlim(0, len(info['mean']) - 1)
        # axes[0].set_ylim(0, limits[dimension])
        axes[1].set_xlim(0, len(info['mean']) - 1)
        # axes[1].set_ylim(0, limits[dimension])

        axes[0].legend(loc='upper left')
        # axes[1].legend(loc='upper left')
        
        axes[0].set_title('Combined Loss', fontsize=20)
        axes[1].set_title('Velocity Loss', fontsize=20)

        plt.savefig(f'visuals/AR_losses/{dimension}_combined_vs_velocity.png', bbox_inches='tight')
        plt.show()
        del fig

        print(f'\n{dimension}')
        df = construct_table(filenames, dimension)
        print(df['AutoRegressive'])
        #save df as LaTeX
        for modtype in ['OneStep', 'AutoRegressive']:
            with open(f'visuals/latex_tables/{modtype}_{dimension}.txt', 'w') as fp:
                latex = df[modtype].to_latex(buf = fp, escape=False)


if __name__ == '__main__':
    main()
