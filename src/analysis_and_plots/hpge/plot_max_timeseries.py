#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | This module creates a 2D field of maximum spurious current |
#     | in the vertical and in time after an HPGE test.            |
#     | The resulting file can be used then to optimise the rmax   |
#     | of Multi-Envelope vertical grids.                          |
#     |                                                            |
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import os
from os.path import join, isfile, basename, splitext
import glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr

# ==============================================================================
# Input files
# ==============================================================================

# Folder path containing HPGE spurious currents velocity files 
MAINdir = '/local/path/to/model_data'
HPGElst = ['zps','vqs','szt','mes']

# Name of the zonal and meridional velocity variables
Uvar = 'uo'
Vvar = 'vo'
# Name of the variable to chunk with dask and size of chunks
chunk_var = 'time_counter'
chunk_size = 1

cols = ["red","blue","dodgerblue","limegreen"]

# ==============================================================================
# OPENING fig
fig, ax = plt.subplots(figsize=(16,9))

# LOOP

for exp in range(len(HPGElst)):

    HPGEdir = MAINdir + HPGElst[exp]

    Ufiles = sorted(glob.glob(HPGEdir+'/*grid-U.nc'))
    Vfiles = sorted(glob.glob(HPGEdir+'/*grid-V.nc'))

    for F in range(len(Ufiles)):

        print(Ufiles[F])

        ds_U = xr.open_dataset(Ufiles[F], chunks={chunk_var:chunk_size})
        U4   = ds_U[Uvar]
        ds_V = xr.open_dataset(Vfiles[F], chunks={chunk_var:chunk_size})
        V4   = ds_V[Vvar]

        # rename some dimensions
        U4 = U4.rename({U4.dims[0]: 't', U4.dims[1]: 'k'})
        V4 = V4.rename({V4.dims[0]: 't', V4.dims[1]: 'k'})

        # interpolating from U,V to T
        U = U4.rolling({'x':2}).mean().fillna(0.)
        V = V4.rolling({'y':2}).mean().fillna(0.) 
    
        hpge = np.sqrt(np.power(U,2) + np.power(V,2))

        max_hpge = hpge.max(dim=('k','y','x'))

        ax.plot(max_hpge.data, linestyle="-", linewidth=5, color=cols[exp], label=HPGElst[exp])
       
plt.rc('legend', **{'fontsize':30})
ax.legend(loc=0, ncol=1, frameon=False)
ax.set_xlabel('Days', fontsize=35)
ax.set_ylabel(r'max $| \mathbf{u} |$ [$m\;s^{-1}$]', fontsize=35)
ax.tick_params(axis='both',which='major', labelsize=30)
ax.set_xlim(0.,30)
ax.set_ylim(0.,0.3)
ax.grid(True)
name = 'max_hpge_timeseries.png'
plt.savefig(name)
