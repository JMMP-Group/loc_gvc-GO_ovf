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
from xnemogcm import open_domain_cfg, open_nemo
import cartopy.crs as ccrs
import cmocean
from utils import plot_bot_plume
from matplotlib import pyplot as plt
import matplotlib.colors as colors

# ==============================================================================
# Input parameters

# 1. INPUT FILES
# Change this to match your local paths set-up
base_dir = "/your_local_path"

exp = "MEs" # "zps", "szt"
DOMCFG_dir = base_dir + '/loc_gvc-nordic_ovf/models_geometry/dom_cfg/realistic/' + exp.lower()
TRACER_dir = base_dir + '/loc_gvc-nordic_ovf/outputs/ideal_ovf/' + exp

DOMCFG_list = [DOMCFG_dir + '/domain_cfg_' + exp.lower() + '.nc']
TRACER_list = [TRACER_dir + '/nemo_cg602o_1d_19760101-19760201_grid_T.nc',
               TRACER_dir + '/nemo_cg602o_1d_19760201-19760301_grid_T.nc',
               TRACER_dir + '/nemo_cg602o_1d_19760301-19760401_grid_T.nc',
               TRACER_dir + '/nemo_cg602o_1d_19760401-19760501_grid_T.nc']

# 2. ANALYSIS
tra_lim = 0.1 # minimum passive tracer [] -> to identify dense plume 

# 3. PLOT
lon0 = -37.5
lon1 = -22.5
lat0 =  61.5
lat1 =  67.5

proj = ccrs.Mercator() #ccrs.Robinson()

# ==============================================================================

# Loading domain geometry
ds_dom  = open_domain_cfg(files=DOMCFG_list)
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        ds_dom[i] = ds_dom[i].rename({dim: dim+"_c"})

# Loading NEMO files
ds_T = open_nemo(ds_dom, files=TRACER_list)


# Extracting only the part of the domain we need

ds_dom = ds_dom.isel(x_c=slice(880,1150),x_f=slice(880,1150),y_c=slice(880,1140),y_f=slice(880,1140)) 
ds_T =  ds_T.isel(x_c=slice(880,1150),y_c=slice(880,1140))

bathy = ds_dom["bathymetry"].isel(x_c=slice(1, None), y_c=slice(1, None))

# Plotting bottom cells overflow ----------------------------------------------------------
lev = ds_dom['bottom_level'].load()-1 # because indexes are in Fortran convention
lev = lev.where(lev>0,0) # we removenegative indexes
da_T_bot = ds_T['so_seos'].isel(z_c=lev).load()

bathy = ds_dom["bathymetry"]
fig_path = "./"
lon = ds_dom["glamf"]
lat = ds_dom["gphif"]
map_lims = [lon0, lon1, lat0, lat1]

cn_level   = [0.1,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.,8.,9.,10]
cmap = plt.get_cmap('jet')
CMAP = colors.ListedColormap(cmap(np.linspace(0,1.,len(cn_level))))

colmap = CMAP
vmin = 0.
vmax = 10.
cbar_extend = "neither"
cbar_label = "Passive tracer conc. [ppt]"
cbar_hor = 'horizontal'
cn_lev = [0., 250., 500., 750., 1000., 1250., 1500, 1750., 2000., 2500., 3000., 3500.]


for t in [30,60,90]:

    da = da_T_bot[t,:,:].squeeze()
    da = da - 20. # environment [] of the passive tracer
    da = da.where(da>tra_lim)
    
    fig_name = "ovf_bottom_"+exp+"_"+str(t)+".png"
    var = da.isel(x_c=slice(1, None), y_c=slice(1, None))

    plot_bot_plume(fig_name, fig_path, lon, lat, var, proj,
                   colmap, cn_level, vmax, cbar_extend, cbar_label, 
                   cbar_hor, map_lims, bathy, cn_lev)
 
