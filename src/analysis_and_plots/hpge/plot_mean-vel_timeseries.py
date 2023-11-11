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

base_dir = "/your_local_path"

# Folder path containing HPGE spurious currents velocity files 
MAINdir = '/local/path/to/model_data'
HPGElst = ['zps','vqs','szt','mes']

loc_msk = base_dir + '/models_geometry/loc_area/bathymetry.loc_area.dep2800_novf_sig1_stn9_itr1.nc'

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

    # Loading domain geometry
    ds_dom  = xr.open_dataset(DOMCFG[exp]).squeeze()

    # Computing land-sea masks
    ds_dom = compute_masks(ds_dom, merge=True)

    ds_dom = ds_dom.isel(x=slice(920,1220),y=slice(905,1140))

    e3t = ds_dom["e3t_0"].squeeze()
    e2t = ds_dom["e2t"].squeeze()
    e1t = ds_dom["e1t"].squeeze()

    e1t = e1t.where(ds_dom.tmask==1)
    e2t = e2t.where(ds_dom.tmask==1)
    e3t = e3t.where(ds_dom.tmask==1)

    e1t = e1t.where(loc_msk==1)
    e2t = e2t.where(loc_msk==1)
    e3t = e3t.where(loc_msk==1)

    HPGEdir = MAINdir + HPGElst[exp]

    Ufiles = sorted(glob.glob(HPGEdir+'/*grid-U.nc'))
    Vfiles = sorted(glob.glob(HPGEdir+'/*grid-V.nc'))

    v_avg = []

    for F in range(len(Ufiles)):

        print(Ufiles[F])

        ds_U = xr.open_dataset(Ufiles[F], chunks={chunk_var:chunk_size})
        ds_V = xr.open_dataset(Vfiles[F], chunks={chunk_var:chunk_size})
        ds_U = ds_U.rename_dims({'depthu':'z'})
        ds_V = ds_V.rename_dims({'depthv':'z'})
        ds_U = ds_U.isel(x=slice(920,1220), y=slice(905,1140))
        ds_V = ds_V.isel(x=slice(920,1220), y=slice(905,1140))
        U4   = ds_U[Uvar]
        V4   = ds_V[Vvar]

        # rename some dimensions
        U4 = U4.rename({U4.dims[0]: 't'})
        V4 = V4.rename({V4.dims[0]: 't'})

        # interpolating from U,V to T
        U = U4.rolling({'x':2}).mean().fillna(0.)
        V = V4.rolling({'y':2}).mean().fillna(0.) 
    
        vel = np.sqrt((np.power(U,2) + np.power(V,2)))
        vel = vel.where(ds_dom.tmask==1)
        vel = vel.where(loc_msk==1)

        cel_vol = e1t * e2t * e3t
        dom_vol = cel_vol.sum(skipna=True)
        v_avg.extend(((cel_vol*vel).sum(dim=["x","y","z"], skipna=True) / dom_vol).values.tolist())

    ax.plot(v_avg, linestyle="-", linewidth=5, color=cols[exp], label=HPGElst[exp])
           
plt.rc('legend', **{'fontsize':30})
ax.legend(loc=0, ncol=1, frameon=False)
ax.set_xlabel('Days', fontsize=35)
ax.set_ylabel(r'max $| \mathbf{u} |$ [$m\;s^{-1}$]', fontsize=35)
ax.tick_params(axis='both',which='major', labelsize=30)
#ax.set_xlim(0.,30)
#ax.set_ylim(0.,0.3)
ax.grid(True)
name = 'avg_vel_timeseries.png'
plt.savefig(name)
