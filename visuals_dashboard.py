import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, json, zlib, os, random, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yaml

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
configs = [('pdebench_2d_ar', 1), ('pdebench_2d_ar_momentum', 0),]

TIMES = [1, 3, 5, 10]
IDX = 0

with st.spinner('Loading...'):
    config_info, pred_files, target_files, report_files = [], [], [], []
    data = defaultdict(list)
    time_series_data = defaultdict()

    for i, (config, use_target) in enumerate(configs):
        loss_data = read_loss_file(f'out/{config}/test_losses.json')
        data['name'].append(config)
        for term, (mean, std) in loss_data.items():
            data[term].append((f'{mean:.3g} +/- {std:.3g}'))
        
        pred_files.append(f'out/{config}/test_preds.npy')
        
        if use_target:
            target_files.append(f'out/{config}/test_targets.npy')
        else:
            target_files.append(None)

        with open(f'configs/{config}.yaml', 'r') as stream:
            config_info.append(yaml.load(stream, yaml.FullLoader))

        report_files.append(read_report_file(f'out/{config}/report.txt'))

    data = pd.DataFrame(data)
    # data = data.set_index('name')
    data = data.sort_index()

file_tags = [config['model'] for config in config_info]
tag_bool = {f'{tag}:{ft[tag]}': True for ft in file_tags for tag in ft}
for tag in tag_bool:
    tag_bool[tag] = st.toggle(tag, value=True)

# filter files
selected_files = []
selected_reports = []
for p_file, t_file, r_file, tags in zip(pred_files, target_files, report_files, file_tags):
    if all(tag_bool[f'{tag}:{tags[tag]}'] for tag in tags):
        selected_files.append((p_file, t_file))
        selected_reports.append((p_file.split('/')[-2], r_file))

st.header('Loss Statistics')
st.dataframe(data)

st.header('Loss Curves')

keys = list(filter(lambda x: 'loss' in x, selected_reports[0][1].keys()))
term = st.selectbox('select term to visualize', options=keys, index=0)

fig = plt.figure(figsize=(10, 6))
for name, r_file in selected_reports:
    if term in r_file:
        plt.plot(r_file[term], label=name)

plt.legend()
plt.xlabel('Epochs')
plt.ylabel(term)

st.pyplot(fig)

st.header('Flow Visualization')

format_fis = {0:'density', 1:'vel_x', 2:'vel_y', 3:'pressure'}
fi = st.selectbox('select feature to visualize', options=range(len(format_fis)), format_func=lambda i: format_fis[i])

fig, axes = plt.subplots(1 + len(selected_files), 6, figsize = (18, 5 + 4 * len(selected_files)), layout='constrained',
                        gridspec_kw=dict(width_ratios=[6,1,6,6,6,6], hspace=0, wspace=0.1), squeeze = False)


for j, (p_file, t_file) in enumerate(selected_files):

    if t_file is not None:
        test_data = np.load(t_file)[IDX]

        vmin, vmax = test_data[...,fi].min(), test_data[...,fi].max()
        for i in range(len(axes)):
            axes[i, 0].imshow(test_data[0,...,fi], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            axes[0, 0].set_title(f"T=0", fontsize=24)

        for i in range(0, len(TIMES)):
            ax = axes[0, i+2]    
            ax.set_title(f'T={TIMES[i]}', fontsize=20)
            ax.imshow(test_data[TIMES[i],...,fi], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[0, 0].set_ylabel('Ground Truth', fontsize=24)
        axes[0, 1].remove()

    name = p_file.split('/')[-2]
    arr = np.load(p_file)[IDX]
    j += 1
    for i, t in enumerate(TIMES):
        ax = axes[j, i+2]
        ax.imshow(arr[t,...,fi], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[j, 0].set_ylabel(name.replace('-', '\n'), fontsize=24)
    axes[j, 1].remove()
    
    fig.suptitle('Flow Progression for 2D CFD', fontsize=28)

st.pyplot(fig)


st.header('Error Visualization')
fig, axes = plt.subplots(1 + len(selected_files), 6, figsize = (18, 5 + 4 * len(selected_files)), layout='constrained',
                        gridspec_kw=dict(width_ratios=[6,1,6,6,6,6], hspace=0, wspace=0.1), squeeze = False)

for j, (p_file, t_file) in enumerate(selected_files):

    if t_file is not None:
        test_data = np.load(t_file)[IDX]

        vmin, vmax = test_data[...,fi].min(), test_data[...,fi].max()
        for i in range(len(axes)):
            axes[i, 0].imshow(test_data[0,...,fi], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            axes[0, 0].set_title(f"T=0", fontsize=24)

        for i in range(0, len(TIMES)):
            ax = axes[0, i+2]    
            ax.set_title(f'T={TIMES[i]}', fontsize=20)
            ax.imshow(test_data[TIMES[i],...,fi], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
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
        error = arr[t,...,fi] - test_data[t,...,fi]
        ims.append(ax.imshow(error, cmap='RdBu', origin='lower'))
        ax.set_xticks([])
        ax.set_yticks([])
        vdeg = max(vdeg, abs(error.max()))

    axes[j, 0].set_ylabel(name.replace('-', '\n'), fontsize=24)
    axes[j, 1].remove()
    
    fig.suptitle('Flow Error Progression for 2D CFD', fontsize=28)

for im in ims:
    im.norm.autoscale([-vdeg, vdeg])

st.pyplot(fig)

# Visualize Fourier modes
# st.header('Fourier Visualization')

# fft = np.fft.rfftn(test_data[hist].transpose(2, 0, 1), axes=(-2, -1))
# print(fft.shape)

# fig, axes = plt.subplots(1, 4)
# for i in range(len(axes)):
#     axes[i].matshow(np.abs(fft[i]))

# st.pyplot(fig)
