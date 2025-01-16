import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys, json, zlib, os, random, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

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

    name = file.split('\\')[-1][:-17]
    losses['name'] = name
    
    return losses

st.title('Constrained Physics Informed Neural Operators - Evaluation')
"""Sean Current"""

# DATAFILE = st.file_uploader('Upload JSON source/target/samples dataset:', accept_multiple_files=True)
# if not DATAFILE:
#     st.stop()

# directory = 'out\\norm_loss'
# directory = 'out\\std_scaled'
# directory = 'out\\std_scaled_norm_loss'
# directory = 'out\\k_varying'
# directory = 'out\\time_dim'
# directory = 'out\\static'
# directory = 'out\\static_debugging'
# directory = 'out\\static_debugging_pino'
# directory = 'out\\up_to_date'
directory = 'out\\fourier_loss'

DATAFILE = os.listdir(directory)

PDEBENCH = True
# PDEBENCH = False

with st.spinner('Loading...'):
    loss_files, test_files = [], []
    for i, file in enumerate(DATAFILE):
        file = os.path.join(directory, file)

        if 'PDEBench' in file and not PDEBENCH:
            continue
        elif 'PINO' in file and PDEBENCH:
            continue

        if file.endswith('.json'):
            loss_files.append(read_loss_file(file))

        elif file.endswith('_test.npy'):
            test_files.append(file)
            
        else:
            pass

    # test_data = np.load('data/GT_test_2D.npy').squeeze()
    if PDEBENCH:
        test_data = np.load('data/GT_test_2D_std_scaled.npy')[0].squeeze()
    else:
        test_data = np.load('data/GT_test_2D_vorticity.npy')[0]
    test_data = np.moveaxis(test_data, -2, 0)
    hist = 0

    data = defaultdict(list)
    for loss_data in loss_files:
        data['name'].append(loss_data.pop('name'))
        for term, (mean, std) in loss_data.items():
            data[term].append(f'{mean:.3g} +/- {std:.3g}')

    data = pd.DataFrame(data)
    # data = data.set_index('name')
    data = data.sort_index()

file_tags = [set(tag for tag in file.split('\\')[-1][:-9].split('-')) for file in test_files]
tag_bool = {tag: True for ft in file_tags for tag in ft}
for tag in tag_bool:
    tag_bool[tag] = st.toggle(tag, value=True)

# filter files
selected_files = [file for file, tags in zip(test_files, file_tags) if all(tag_bool[tag] for tag in tags)]

st.header('Loss Statistics')
# st.dataframe(data.loc[[file.split('\\')[-1][:-9] for file in selected_files]])

st.header('Flow Visualization')

format_fis = {0:'density', 1:'vel_x', 2:'vel_y', 3:'pressure'}
fi = st.selectbox('select feature to visualize', options=range(len(format_fis)), format_func=lambda i: format_fis[i])

# selected_files = st.multiselect('select flows to visualize', range(len(test_files)),
                                # format_func=lambda i: test_files[i].split('\\')[-1][:-9])


fig, axes = plt.subplots(1 + len(selected_files), 6, figsize = (18, 5 + 4 * len(selected_files)), layout='constrained',
                        gridspec_kw=dict(width_ratios=[6,1,6,6,6,6], hspace=0, wspace=0.1), squeeze = False)
if PDEBENCH:
    times = [1, 3, 5, 10]
else:
    times = [1, 2, 5, 20]

# GT first
vmin, vmax = test_data[...,fi].min(), test_data[...,fi].max()
for i in range(len(axes)):
    axes[i, 0].imshow(test_data[hist,...,fi], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])

    axes[0, 0].set_title(f"T=0", fontsize=24)

for i in range(0, len(times)):
    ax = axes[0, i+2]    
    ax.set_title(f'T={times[i]}', fontsize=20)
    ax.imshow(test_data[times[i] + hist,...,fi], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

axes[0, 0].set_ylabel('Ground Truth', fontsize=24)
axes[0, 1].remove()

for j, file in enumerate(selected_files):
    name = selected_files[j].split('\\')[-1][:-9]
    arr = np.load(file)
    arr = arr.squeeze(1)
    j += 1
    for i, t in enumerate(times):
        ax = axes[j, i+2]
        ax.imshow(arr[t-1,...,fi], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[j, 0].set_ylabel(name.replace('-', '\n'), fontsize=24)
    axes[j, 1].remove()
    
    fig.suptitle('Flow Progression for 2D CFD', fontsize=28)

st.pyplot(fig)

st.header('Error Visualization')

# selected_files = st.multiselect('select flows to visualize', range(len(test_files)),
                                # format_func=lambda i: test_files[i].split('\\')[-1][:-9])


fig, axes = plt.subplots(1 + len(selected_files), 6, figsize = (18, 5 + 4 * len(selected_files)), layout='constrained',
                        gridspec_kw=dict(width_ratios=[6,1,6,6,6,6], hspace=0, wspace=0.1), squeeze = False)

# GT first
vmin, vmax = test_data[...,fi].min(), test_data[...,fi].max()
for i in range(len(axes)):
    axes[i, 0].imshow(test_data[hist,...,fi], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])

    axes[0, 0].set_title(f"T=0", fontsize=24)

for i in range(0, len(times)):
    ax = axes[0, i+2]    
    ax.set_title(f'T={times[i]}', fontsize=20)
    ax.imshow(test_data[hist+times[i],...,fi], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

axes[0, 0].set_ylabel('Ground Truth', fontsize=24)
axes[0, 1].remove()

vdeg = 0
ims = []
for j, file in enumerate(selected_files):
    name = selected_files[j].split('\\')[-1][:-9]
    arr = np.load(file).squeeze()
    j += 1
    for i, t in enumerate(times):
        ax = axes[j, i+2]
        error = arr[t-1,...,fi] - test_data[hist+t,...,fi]
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
