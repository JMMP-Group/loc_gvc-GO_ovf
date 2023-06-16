#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xarray as xr
from xnemogcm import open_domain_cfg, open_nemo
from geopy.distance import great_circle
import matplotlib.colors as colors
from scipy.stats import binned_statistic_2d

# ==============================================================================
# Input parameters

# 1. INPUT FILES

# Change this to match your local paths set-up
base_dir = "/your_local_path"

exp_list = ["zps", "szt", "MEs"]

comm_dir = base_dir + '/loc_gvc-nordic_ovf/outputs/ideal_ovf'

File = "_dia_mixing_time_4d.nc"

# ==============================================================================

for exp in range(len(exp_list)):

    print(exp_list[exp])

    filename = comm_dir + exp_list[exp] + '/' + exp_list[exp] + File 

    # Loading file
    ds  = xr.open_dataset(filename)
    H   = ds.H * 1.e-3 # in kg
    time = ds.time.values
    density = ds.density.values - 1000. # density anomaly
    time = np.insert(time, 0, 0.)
    density = np.insert(density, 0, 27.6)

    H = np.ma.masked_where(H==0., H) #masking where there is no data
    #print(H[0,:].sum())
    #H = H / H[0,:].sum()
    XX, YY = np.meshgrid(time, density)

    fig = plt.figure(figsize=(13, 10))
    ax = plt.subplot(111)
    #pc = ax.pcolormesh(XX,YY,H.T,cmap='inferno_r', vmin=0., vmax=1.e12)
    pc = ax.pcolormesh(XX,YY,H.T,cmap='inferno_r', vmin=0., vmax=4.e13)
    ax.set_ylim(27.66, 28.74)
    ax.invert_yaxis()
    plt.xticks(time)
    plt.yticks(density)

    for n, labely in enumerate(ax.yaxis.get_ticklabels()):
        if n % 3 != 0: labely.set_visible(False)
    for n, labelx in enumerate(ax.xaxis.get_ticklabels()):
        if n % 5 != 0: labelx.set_visible(False)

    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)

    if exp == 1:
       # Internal colorbar
       cbaxes = inset_axes(ax, width="42%", height="4.2%", loc=4, borderpad=2.7)
       #cb = plt.colorbar(pc, cax=cbaxes, ticks=[0., 0.5e12, 1.e12], orientation='horizontal', extend="max")
       cb = plt.colorbar(pc, cax=cbaxes, ticks=[0., 2.e13, 4.e13], orientation='horizontal', extend="max")
       cb.ax.tick_params(color='black', labelcolor='black', labelsize=20, width=3, length=10, direction='in')
       cb.ax.xaxis.get_offset_text().set_fontsize(20)
       cb.ax.xaxis.get_offset_text().set_position((1.1,0.6))
       cb.ax.xaxis.OFFSETTEXTPAD = -75
       cb.outline.set_edgecolor('black')
       cb.formatter.set_useMathText(True)
       #cb.set_label(label="[g]",size=20, color='black')

    #if exp == 3:
    #   # External colorbar
    #   cb = plt.colorbar(pc, 
    #                     ax=ax, 
    #                     pad=.2, 
    #                     orientation='horizontal', 
    #                     extend="max")
    #   cb.ax.tick_params(color='black', labelcolor='black', labelsize=20)
    #   cb.ax.xaxis.get_offset_text().set_fontsize(20)
    #   cb.ax.xaxis.get_offset_text().set_position((1.05,0.6))
    #   cb.ax.xaxis.OFFSETTEXTPAD = -90
    #   cb.outline.set_edgecolor('black')
    #   cb.formatter.set_useMathText(True)
    #   cb.set_label(label="Tot. amount of tracer [kg]",size=20, color='black')

    ax.grid(visible=True, color='black')

    if exp < 3:
       plt.savefig(exp_list[exp] + '_dia_mixing_time.png', bbox_inches="tight")
    else:
       plt.savefig('colorbar_dia_mixing_time.png', bbox_inches="tight")
    print("done")
    plt.close()
