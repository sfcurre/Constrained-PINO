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

st.title('Uncertain Physics Informed Neural Operators - Evaluation')
"""Sean Current"""

directory = 'out\\uncertainty'
DATAFILE = os.listdir(directory)

with st.spinner('Loading...'):
    test_files = []
    for i, file in enumerate(DATAFILE):
        file = os.path.join(directory, file)

        if file.endswith('_test.npy'):
            test_files.append(file)
            
    test_data = np.load('data/GT_test_2D_vorticity.npy')[0]
    test_data = np.moveaxis(test_data, -2, 0)
    hist = 0

file_tags = [set(tag for tag in file.split('\\')[-1][:-9].split('-')) for file in test_files]
tag_bool = {tag: True for ft in file_tags for tag in ft}
for tag in tag_bool:
    tag_bool[tag] = st.toggle(tag, value=True)

# filter files
selected_files = [file for file, tags in zip(test_files, file_tags) if all(tag_bool[tag] for tag in tags)]

st.header('Flow Visualization')

times = [1, 3, 5, 10, 20, 40]
fig, axes = plt.subplots(1 + len(selected_files) * 2, 2 + len(times), figsize = (6 + 3 * len(times), 5 + 4 * len(selected_files)), layout='constrained', gridspec_kw=dict(width_ratios=[6,1] + [6] * len(times), hspace=0, wspace=0.1), squeeze = False)

# GT first
vmin, vmax = test_data[...,0].min(), test_data[...,0].max()
for i in range(len(axes)):
    axes[i, 0].imshow(test_data[hist,...,0], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])

    axes[0, 0].set_title(f"T=0", fontsize=24)

for i in range(0, len(times)):
    ax = axes[0, i+2]    
    ax.set_title(f'T={times[i]}', fontsize=20)
    ax.imshow(test_data[times[i] + hist,...,0], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

axes[0, 0].set_ylabel('Ground Truth', fontsize=24)
axes[0, 1].remove()

j = 0
for file in selected_files:
    name = file.split('\\')[-1][:-9]
    arr = np.load(file)
    arr = arr.squeeze(1)
    
    j += 1
    for i, t in enumerate(times):
        ax = axes[j, i+2]
        ax.imshow(arr[t-1,...,0], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    
    axes[j, 0].set_ylabel(name.replace('-', '\n'), fontsize=24)
    axes[j, 1].remove()

    j += 1
    for i, t in enumerate(times):
        ax = axes[j, i+2]
        ax.imshow(np.sqrt(arr[t-1,...,-1]), cmap='Reds', origin='lower', vmin=0, vmax=np.sqrt(arr[...,-1]).max())
        ax.set_xticks([])
        ax.set_yticks([])

    axes[j, 0].set_ylabel(name.replace('-', '\n') + ' stdev', fontsize=24)
    axes[j, 1].remove()
    
    fig.suptitle('Flow Progression for 2D CFD', fontsize=28)

st.pyplot(fig)

st.header('Error Visualization')

# selected_files = st.multiselect('select flows to visualize', range(len(test_files)),
                                # format_func=lambda i: test_files[i].split('\\')[-1][:-9])

fig, axes = plt.subplots(1 + len(selected_files) * 2, 2 + len(times), figsize = (6 + 3 * len(times), 5 + 4 * len(selected_files)), layout='constrained', gridspec_kw=dict(width_ratios=[6,1] + [6] * len(times), hspace=0, wspace=0.1), squeeze = False)

# GT first
vmin, vmax = test_data[...,0].min(), test_data[...,0].max()
for i in range(len(axes)):
    axes[i, 0].imshow(test_data[hist,...,0], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])

    axes[0, 0].set_title(f"T=0", fontsize=24)

for i in range(0, len(times)):
    ax = axes[0, i+2]    
    ax.set_title(f'T={times[i]}', fontsize=20)
    ax.imshow(test_data[hist+times[i],...,0], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

axes[0, 0].set_ylabel('Ground Truth', fontsize=24)
axes[0, 1].remove()

vdeg = 0
ims = []
j = 0
for file in selected_files:
    name = file.split('\\')[-1][:-9]
    arr = np.load(file).squeeze()
    j += 1
    for i, t in enumerate(times):
        ax = axes[j, i+2]
        error = arr[t-1,...,0] - test_data[hist+t,...,0]
        ims.append(ax.imshow(error, cmap='RdBu', origin='lower'))
        ax.set_xticks([])
        ax.set_yticks([])
        vdeg = max(vdeg, abs(error.max()))

    axes[j, 0].set_ylabel(name.replace('-', '\n'), fontsize=24)
    axes[j, 1].remove()

    j += 1
    for i, t in enumerate(times):
        ax = axes[j, i+2]
        ax.imshow(np.sqrt(arr[t-1,...,-1]), cmap='Reds', origin='lower', vmin=0, vmax=np.sqrt(arr[...,-1]).max())
        ax.set_xticks([])
        ax.set_yticks([])

    axes[j, 0].set_ylabel(name.replace('-', '\n') + ' stdev', fontsize=24)
    axes[j, 1].remove()
    
    fig.suptitle('Flow Error Progression for 2D CFD', fontsize=28)

for im in ims:
    im.norm.autoscale([-vdeg, vdeg])

st.pyplot(fig)

