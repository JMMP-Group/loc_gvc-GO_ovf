#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import os
from os.path import join, isfile, basename, splitext
import glob
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from xnemogcm import open_domain_cfg
import cartopy.crs as ccrs
import cmocean
from utils import plot_hpge

# ==============================================================================
# Input parameters

# 1. INPUT FILES

# Change this to match your local paths set-up
base_dir = "/your_local_path"

vcoord = ['zps', 'vqs', 'szt', 'mes'] 

# 3. PLOT
lon0 = -45.
lon1 =  5.0
lat0 =  50.
lat1 =  72.
proj = ccrs.Mercator() #ccrs.Robinson()

# ==============================================================================

for vco in range(len(vcoord)):

    HPGE_dir = base_dir + '/hpge/' + vcoord[vco]
    if vcoord[vco] == 'vqs': HPGE_dir = HPGE_dir + "/r12-r075-r004_v3"
    if vcoord[vco] == 'mes': HPGE_dir = HPGE_dir + "/r12_r12-r075-r040_v3"
    HPGE_file = HPGE_dir + "/maximum_hpge.nc"

    DOMCFG_dir = base_dir + '/models_geometry/dom_cfg/realistic/' + vco
    DOMCFG_file = DOMCFG_dir + '/domain_cfg_' + vco + '.nc'

    # Loading domain geometry
    ds_dom  = open_domain_cfg(files=[DOMCFG_file[vco]])
    for i in ['bathymetry','bathy_meter']:
        for dim in ['x','y']:
            ds_dom[i] = ds_dom[i].rename({dim: dim+"_c"})

    ds_hpge  = xr.open_dataset(HPGE_file)

    # Extracting only the part of the domain we need

    ds_dom  = ds_dom.isel(x_c=slice(880,1200),x_f=slice(880,1200),y_c=slice(880,1140),y_f=slice(880,1140))
    ds_hpge =  ds_hpge.isel(x=slice(880,1200),y=slice(880,1140))

    # Plotting ----------------------------------------------------------

    bathy = ds_dom["bathymetry"]#.isel(x_c=slice(1, None), y_c=slice(1, None))
    varss = list(ds_hpge.keys())

    for env in range(len(varss)):

        if vco == 4:
           fig_name = 'colorbar.png'
        else:
           fig_name = varss[env] + '_' + vcoord[vco] + '.png'

        fig_path = "./"
        lon = ds_dom["glamf"]
        lat = ds_dom["gphif"]
        var = ds_hpge[varss[env]] * 100. # in cm/s 
        colmap = 'hot' #cmocean.cm.ice
        vmin = 0.0
        vmax = 5.
        cbar_extend = 'max' #"max"
        #if vco == 4:
        if vco == 1:
           cbar_label = r"$\times 10^{-2}$ [$m\;s^{-1}$]"
        else:
           cbar_label = ""
        cbar_hor = 'horizontal'
        map_lims = [lon0, lon1, lat0, lat1]
        cn_lev = [0., 250., 500., 1000., 1500, 2000., 3000., 4000.]

        plot_hpge(fig_name, fig_path, lon, lat, var, proj, colmap, 
                  vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, bathy, cn_lev)

 
