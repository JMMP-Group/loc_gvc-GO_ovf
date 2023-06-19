#!/usr/bin/env python

import os
import sys
import subprocess
import numpy as np
import xarray as xr
from xnemogcm import open_domain_cfg
from plot_section import mpl_sec_loop
from utils import compute_masks

# ========================================================================
# INPUT PARAMETERS

# Change this to match your local paths set-up
base_dir = "/your_local_path"
DOMCFG_MEs = base_dir + '/models_geometry/dom_cfg/realistic/mes/domain_cfg_mes.nc'
BATHY_MEs = base_dir + '/models_geometry/loc_area/bathymetry.loc_area.dep2800_novf_sig1_stn9_itr1.nc'

# 2. ANALYSIS

# Iceland-Faroe Ridge
sec_lon1 = [ 0.34072625, -3.56557722,-18.569585  ,-26.42872351, -30.314948]
sec_lat1 = [68.26346438, 65.49039963, 60.79252542, 56.24488972,  52.858934]

# Denmark Strait
sec_lon2 = [-10.84451672, -25.30818606, -35., -44.081319]
sec_lat2 = [ 71.98049514,  66.73449533,  61.88833838,  56.000932]

# Iceland-Faroe Ridge
sec_lon1 = [ 0.34072625, -3.56557722,-18.569585  ,-26.42872351, -30.314948]
sec_lat1 = [68.26346438, 65.49039963, 60.79252542, 56.24488972,  52.858934]

# Denmark Strait
sec_lon2 = [-10.84451672, -25.30818606, -35., -44.081319]
sec_lat2 = [ 71.98049514,  66.73449533,  61.88833838,  56.000932]

# For Iceland-Faroe Ridge section:
sec_I_indx_1b_L  = [sec_lon1]
sec_J_indx_1b_L  = [sec_lat1]
xlim_1b_L        = [250, 2500]
# For Denmark Strait section:
#sec_I_indx_1b_L  = [sec_lon2]
#sec_J_indx_1b_L  = [sec_lat2]
#xlim_1b_L        = [0, 2000]

coord_type_1b_L  = "dist"
rbat2_fill_1b_L  = "false"
ylim_1b_L        = [0., 5900.]
vlevel_1b_L      = 'MES'
xgrid_1b_L       = "false"

# ========================================================================
# Reading local-MEs mask
msk_mes = None
ds_msk = xr.open_dataset(BATHY_MEs)
ds_msk = ds_msk.isel(x=slice(880,1150),y=slice(880,1140))
if "s2z_msk" in ds_msk.variables:
   msk_mes = ds_msk["s2z_msk"].values
   #msk_mes[msk_mes>0] = 1
hbatt = []
nenv = 1
while nenv > 0:
  name_env = "hbatt_"+str(nenv)
  if name_env in ds_msk.variables:
      hbatt.append(ds_msk[name_env].values)
      nenv+=1
  else:
      nenv=0
del ds_msk

if msk_mes is not None:
   for env in hbatt:
       env[msk_mes < 2] = np.nan
msk_mes[msk_mes>0] = 1
#msk_mes[msk_mes<2] = 0
#msk_mes[msk_mes>0] = 1

# Loading domain geometry
ds_dom  = open_domain_cfg(files=[DOMCFG_MEs])
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        ds_dom[i] = ds_dom[i].rename({dim: dim+"_c"})

# Computing masks
ds_dom = compute_masks(ds_dom, merge=True)

# Extracting only the part of the domain we need
ds_dom = ds_dom.isel(x_c=slice(880,1150),x_f=slice(880,1150),y_c=slice(880,1140),y_f=slice(880,1140))

tlon2 = ds_dom["glamt"].values
tlat2 = ds_dom["gphit"].values
e3t_3 = ds_dom["e3t_0"].values
e3w_3 = ds_dom["e3w_0"].values
tmsk3 = ds_dom["tmask"].values
bathy = ds_dom["bathymetry"].values

nk = e3t_3.shape[0]
nj = e3t_3.shape[1]
ni = e3t_3.shape[2]

tlon3 = np.repeat(tlon2[np.newaxis, :, :], nk, axis=0)
tlat3 = np.repeat(tlat2[np.newaxis, :, :], nk, axis=0)

# Computing model levels' depth
tdep3 = np.zeros(shape=(nk,nj,ni))
wdep3 = np.zeros(shape=(nk,nj,ni))
wdep3[0,:,:] = 0.
tdep3[0,:,:] = 0.5 * e3w_3[0,:,:]
for k in range(1, nk):
    wdep3[k,:,:] = wdep3[k-1,:,:] + e3t_3[k-1,:,:]
    tdep3[k,:,:] = tdep3[k-1,:,:] + e3w_3[k,:,:]

proj = []

# PLOTTING VERTICAL DOMAIN

var_strng  = ""
unit_strng = ""
date       = ""
timeres_dm = ""
timestep   = []
PlotType   = ""
var4       = []
#hbatt      = []
mbat_ln    = "false"
mbat_fill  = "true"
varlim     = "no"
check      = 'false'
check_val  = 'false'


mpl_sec_loop('ORCA025-locMEs mesh', '.png', var_strng, unit_strng, date, timeres_dm, timestep, PlotType,
              sec_I_indx_1b_L, sec_J_indx_1b_L, tlon3, tlat3, tdep3, wdep3, tmsk3, var4, proj,
              coord_type_1b_L, vlevel_1b_L, bathy, hbatt, rbat2_fill_1b_L, mbat_ln, mbat_fill,
              xlim_1b_L, ylim_1b_L, varlim, check, check_val, xgrid_1b_L, msk_mes=msk_mes)


