import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, json, zlib, os, random, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yaml

from scipy.signal import welch
from skimage.measure import block_reduce

from collections import defaultdict, Counter
from altair import datum

def read_loss_file(file):
    with open(file) as fp:
        data = json.load(fp)

    losses = defaultdict(list)
    for sample in data:
        for term in sample:
            losses[term].append(sample[term])

    for term in losses:
        arr = np.array(losses[term])
        losses[term] = arr.mean(), arr.std()

    return losses

def read_report_file(file):
    data = defaultdict(list)
    with open(file, 'r') as fp:
        for line in fp:
            if line.startswith('Epoch'):
                parts = line.split(',')
                for part in parts:
                    key, val = part.split(':')
                    val = float(val.strip())
                    data[key].append(val)
    return data

st.title('Constrained Physics Informed Neural Operators - Evaluation')
"""Sean Current"""

# configs = [('debug_2d', 1),]
# configs = [('debug_numpy_2d', 1)]
configs = [('pdebench_2d_ar', 1),
           ('pdebench_2d_ar_mass', 0),
           ('pdebench_2d_ar_conservative', 0),
           ('pdebench_2d_ar_momentum', 0),
           ('pdebench_2d_ar_dpot', 0),
           ('pdebench_2d_ar_dpot_mass', 0),
           ('pdebench_2d_ar_dpot_conservative', 0),
           ('pdebench_2d_ar_dpot_momentum', 0),
           ('pdebench_2d_ar_dpot_Ti', 0),
           ('pdebench_2d_ar_dpot_Ti_mass', 0),
           ('pdebench_2d_ar_dpot_Ti_conservative', 0),
           ('pdebench_2d_ar_dpot_Ti_momentum', 0),
           ('pdebench_2d_ar_dpot_S', 0),
           ('pdebench_2d_ar_dpot_S_mass', 0),
           ('pdebench_2d_ar_dpot_S_conservative', 0),
           ('pdebench_2d_ar_dpot_S_momentum', 0),
        ]
# configs = [('pdebench_2d_ar_turb', 1),
#            ('pdebench_2d_ar_mass_turb', 0),
#            ('pdebench_2d_ar_momentum_turb', 0),
#            ('pdebench_2d_ar_turb_nograd', 0),
#            ]
#            ('pdebench_2d_dilated', 0),
#            ('pdebench_2d_dilated_momentum', 0),
#            ]

TIMES = [1, 3, 5, 10, -1]
CMAP = 'jet'

with st.spinner('Loading...'):
    config_info = []
    loss_data = defaultdict(list)

    for i, (config, use_target) in enumerate(configs):
        data = {}
        with open(f'configs/{config}.yaml', 'r') as stream:
            data['config'] = yaml.load(stream, yaml.FullLoader)
            
        loss_file_data = read_loss_file(f'out/{config}/test_losses.json')
        loss_data['name'].append(config)
        for term, (mean, std) in loss_file_data.items():
            loss_data[term].append((f'{mean:.3g} +/- {std:.3g}'))
        
        data['preds'] = f'out/{config}/test_preds.npy'
        
        if os.path.exists(f'out/{config}/test_preds_long.npy'):
            data['long'] = f'out/{config}/test_preds_long.npy'
        else:
            data['long'] =  None

        if use_target:
            data['test'] = f'out/targets/test_targets.npy'
            data['test_long'] = f'out/targets/test_targets_long.npy'
        else:
            data['test'] = None
            data['test_long'] = None

        data['report'] = read_report_file(f'out/{config}/report.txt')
        data['name'] = config

        config_info.append(data)

    loss_data = pd.DataFrame(loss_data)
    loss_data = loss_data.sort_index()

file_tags = [config['config']['model'] for config in config_info]
tag_bool = {f'{tag}:{ft[tag]}': True for ft in file_tags for tag in ft}
for tag in tag_bool:
    tag_bool[tag] = st.toggle(tag, value=True)

# filter files
selected_files = []
for data, tags in zip(config_info, file_tags):
    if all(tag_bool[f'{tag}:{tags[tag]}'] for tag in tags):
        selected_files.append(data)
        
st.header('Loss Statistics')
st.dataframe(loss_data)

st.header('Loss Curves')

keys = list(filter(lambda x: 'loss' in x, selected_files[0]['report'].keys()))
term = st.selectbox('select term to visualize', options=keys, index=0)

fig = plt.figure(figsize=(10, 6))
for data in selected_files:
    r_file = data['report']
    if term in r_file:
        # weights = np.ones(100) / 100
        weights = np.ones(1)
        sma = np.convolve(r_file[term], weights, mode='valid')
        plt.plot(sma, label=data['name'])

plt.legend()
plt.xlabel('Epochs')
plt.ylabel(term)
# plt.ylim(0, 20)

st.pyplot(fig)

st.header('Flow Visualization')

format_fis = {0:'density', 1:'pressure', 2:'vel_x', 3:'vel_y'}
fi = st.selectbox('select feature to visualize', options=range(len(format_fis)), format_func=lambda i: format_fis[i])

IDX = st.selectbox('select sample to visualize', options=range(10), format_func=lambda i: i + 1)

fig, axes = plt.subplots(1 + len(selected_files), 2 + len(TIMES), figsize = (5 * (1 + len(TIMES)), 5 + 4 * len(selected_files)), layout='constrained', gridspec_kw=dict(width_ratios=[6,1] + [6] * len(TIMES), hspace=0, wspace=0.1), squeeze = False)

for j, data in enumerate(selected_files):
    p_file = data['preds']
    t_file = data['test']

    if t_file is not None:
        test_data = np.load(t_file)[IDX]

        vmin, vmax = test_data[...,fi].min(), test_data[...,fi].max()
        for i in range(len(axes)):
            axes[i, 0].imshow(test_data[0,...,fi], cmap=CMAP, origin='lower', vmin=vmin, vmax=vmax)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            axes[0, 0].set_title(f"T=0", fontsize=24)

        for i in range(0, len(TIMES)):
            ax = axes[0, i+2]    
            ax.set_title(f'T={TIMES[i]}', fontsize=20)
            ax.imshow(test_data[TIMES[i],...,fi], cmap=CMAP, origin='lower', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[0, 0].set_ylabel('Ground Truth', fontsize=24)
        axes[0, 1].remove()

    name = p_file.split('/')[-2]
    arr = np.load(p_file)[IDX]
    j += 1
    for i, t in enumerate(TIMES):
        ax = axes[j, i+2]
        ax.imshow(arr[t,...,fi], cmap=CMAP, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[j, 0].set_title(name.replace('-', '\n'), fontsize=24)
    axes[j, 1].remove()
    
    fig.suptitle('Flow Progression for 2D CFD', fontsize=28)

st.pyplot(fig)


st.header('Error Visualization')
fig, axes = plt.subplots(1 + len(selected_files), 2 + len(TIMES), figsize = (5 * (1 + len(TIMES)), 5 + 4 * len(selected_files)), layout='constrained', gridspec_kw=dict(width_ratios=[6,1] + [6] * len(TIMES), hspace=0, wspace=0.1), squeeze = False)

for j, data in enumerate(selected_files):
    p_file = data['preds']
    t_file = data['test']
    
    if t_file is not None:
        test_data = np.load(t_file)[IDX]

        vmin, vmax = test_data[...,fi].min(), test_data[...,fi].max()
        for i in range(len(axes)):
            axes[i, 0].imshow(test_data[0,...,fi], cmap=CMAP, origin='lower', vmin=vmin, vmax=vmax)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            axes[0, 0].set_title(f"T=0", fontsize=24)

        for i in range(0, len(TIMES)):
            ax = axes[0, i+2]    
            ax.set_title(f'T={TIMES[i]}', fontsize=20)
            ax.imshow(test_data[TIMES[i],...,fi], cmap=CMAP, origin='lower', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[0, 0].set_ylabel('Ground Truth', fontsize=24)
        axes[0, 1].remove()

    vdeg = 0
    ims = []
    name = p_file.split('/')[-2]
    arr = np.load(p_file)[IDX]
    j += 1
    for i, t in enumerate(TIMES):
        ax = axes[j, i+2]
        try:
            error = arr[t,...,fi] - test_data[t,...,fi]
            ims.append(ax.imshow(error, cmap='RdBu', origin='lower'))
            ax.set_xticks([])
            ax.set_yticks([])
            vdeg = max(vdeg, abs(error.max()))
        except ValueError:
            ax.remove()

    axes[j, 0].set_title(name.replace('-', '\n'), fontsize=24)
    axes[j, 1].remove()
    
    fig.suptitle('Flow Error Progression for 2D CFD', fontsize=28)

for im in ims:
    im.norm.autoscale([-vdeg, vdeg])

st.pyplot(fig)

st.header('Spectral Density')

fig, axes = plt.subplots(len(selected_files), 3, layout='constrained', figsize = (12, 4 * len(selected_files)), squeeze = True)
for j, data in enumerate(selected_files):
    parr = np.load(data['preds'])[IDX,...,fi]
    if data['test'] is not None:
        tarr = np.load(data['test'])[IDX,...,fi]

    for d in range(3):
        dims = tuple(set((0,1,2)) - {d})
        tfreq, ttransform = welch(tarr, axis=d)
        pfreq, ptransform = welch(parr, axis=d)
        axes[j, d].semilogy(tfreq, ttransform.mean(axis=dims), label='target', color='k')
        axes[j, d].semilogy(pfreq, ptransform.mean(axis=dims), label='pred', color='r', linestyle='--')
        if data['long'] is not None:
            larr = np.load(data['long'])[IDX,...,fi]
            lfreq, ltransform = welch(larr, axis=d)
            axes[j, d].semilogy(lfreq, ltransform.mean(axis=dims), label='long', color='b', linestyle=':')

    axes[j, 0].set_title(data['name'].replace('-', '\n'), fontsize=12)
    axes[j, 0].legend()
    
st.pyplot(fig)


st.header('Long Rollouts')

LONG_TIMES = [10, 20, 30, 50, 75, -1]
selected_longs = [data for data in selected_files if data['long'] is not None]

lfi = st.selectbox('select feature to visualize', options=range(len(format_fis)), format_func=lambda i: format_fis[i], key='lfi')

LIDX = st.selectbox('select sample to visualize', options=range(10), format_func=lambda i: i + 1, key='LIDX')

fig, axes = plt.subplots(1 + len(selected_longs), 2 + len(LONG_TIMES), figsize = (5 * (1 + len(LONG_TIMES)), 5 + 4 * len(selected_longs)), layout='constrained', gridspec_kw=dict(width_ratios=[6,1] + [6] * len(LONG_TIMES), hspace=0, wspace=0.1), squeeze = False)

for j, data in enumerate(selected_files):
    l_file = data['long']
    lt_file = data['test_long']

    if lt_file is not None:
        test_data = np.load(lt_file)[LIDX]

        vmin, vmax = test_data[...,lfi].min(), test_data[...,lfi].max()
        for i in range(len(axes)):
            axes[i, 0].imshow(test_data[0,...,lfi], cmap=CMAP, origin='lower', vmin=vmin, vmax=vmax)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            axes[0, 0].set_title(f"T=0", fontsize=24)

        for i in range(0, len(LONG_TIMES)):
            ax = axes[0, i+2]    
            ax.set_title(f'T={LONG_TIMES[i]}', fontsize=20)
            ax.imshow(test_data[LONG_TIMES[i],...,lfi], cmap=CMAP, origin='lower', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[0, 0].set_ylabel('Ground Truth', fontsize=24)
        axes[0, 1].remove()

    name = l_file.split('/')[-2]
    try:
        arr = np.load(l_file)[LIDX]
    except IndexError:
        continue
    j += 1
    for i, t in enumerate(LONG_TIMES):
        ax = axes[j, i+2]
        ax.imshow(arr[t,...,lfi], cmap=CMAP, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[j, 0].set_title(name.replace('-', '\n'), fontsize=24)
    axes[j, 1].remove()
    
    fig.suptitle('Long Rollout for 2D CFD', fontsize=28)

st.pyplot(fig)


st.header('Reduction Visualization')

test_data = np.load(selected_files[0]['test'])[IDX]
fig, axes = plt.subplots(4, len(TIMES) + 2, figsize = (20, 20), layout='constrained')

vis_data = np.zeros_like(test_data)
vis_data[...,0] = test_data[...,0]
vis_data[...,1] = test_data[...,1] * (1 / (5/3 - 1)) + 0.5 * (test_data[...,2]**2 + test_data[...,3]**2) * test_data[...,0]
vis_data[...,2] = test_data[...,2] * test_data[...,0]
vis_data[...,3] = test_data[...,3] * test_data[...,0] 
for i in range(4):
    
    ax = axes[i, 0]
    ax.imshow(vis_data[0,...,fi], cmap=CMAP, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_title(f"T=0", fontsize=20)
    ax.set_ylabel(vis_data.shape[1:-1], fontsize=20)

    for j, t in enumerate(TIMES):
        ax = axes[i, j+1]
        if i == 0:
            ax.set_title(f'T={t}', fontsize=20)
        ax.imshow(vis_data[t,...,fi], cmap=CMAP, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])

    # conservations
    total_density = vis_data[...,0].sum(axis=(1,2))
    total_pressure = vis_data[...,1].sum(axis=(1,2))
    total_vx = vis_data[...,2].sum(axis=(1,2))
    total_vy = vis_data[...,3].sum(axis=(1,2))
    axes[i, -1].plot(total_density - total_density.mean(), label='density')
    axes[i, -1].plot(total_pressure - total_pressure.mean(), label='energy')
    axes[i, -1].plot(total_vx - total_vx.mean(), label='mx')
    axes[i, -1].plot(total_vy - total_vy.mean(), label='my')
    if i == 0:
        axes[i, -1].set_title('Conservations', fontsize=20)
    axes[i, -1].legend(fontsize=16)

    vis_data = block_reduce(vis_data, block_size=(1,2,2,1), func=np.mean)
    # vis_data = vis_data[:, ::2, ::2, :]

st.pyplot(fig)