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
import matplotlib.colors as colors
import xarray as xr
from xnemogcm import open_domain_cfg, open_nemo
import cartopy.crs as ccrs
import cmocean
import gsw as gsw
from utils import plot_bot_plume
import scipy.interpolate as interpolate

# ------------------------------------------------------------------------------------------
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
# -------------------------------------------------------------------------------------------
class TwoInnerPointsNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
        self.low = low
        self.up = up
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.25, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# ==============================================================================

# Input parameters

# 1. INPUT FILES

# Change this to match your local paths set-up
base_dir = "/your_local_path"

DOMCFG_zps = base_dir + '/loc_gvc-nordic_ovf/models_geometry/dom_cfg/realistic/zps/domain_cfg_zps.nc'
DOMCFG_szt = base_dir + '/loc_gvc-nordic_ovf/models_geometry/dom_cfg/realistic/szt/domain_cfg_szt.nc'
DOMCFG_MEs = base_dir + '/loc_gvc-nordic_ovf/models_geometry/dom_cfg/realistic/mes/domain_cfg_mes.nc'

Tzpsdir = base_dir + '/loc_gvc-nordic_ovf/outputs/realistic/zps'
Tsztdir = base_dir + '/loc_gvc-nordic_ovf/outputs/realistic/szt'
Tmesdir = base_dir + '/loc_gvc-nordic_ovf/outputs/realistic/MEs'

# 3. PLOT
# 3. PLOT
lon0 = -61. # -47.
lon1 = -5.  # -2..
lat0 =  46. # 50.
lat1 =  67. # 72.
proj = ccrs.Mercator() #ccrs.Robinson()

fig_path = "./"
cbar_extend = "both"
cn_lev = [500., 1000., 2000., 3000.]
cbar_hor = 'horizontal'
map_lims = [lon0, lon1, lat0, lat1]

# ==============================================================================

# Loading domain geometry
DS_zps  = open_domain_cfg(files=[DOMCFG_zps])
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        DS_zps[i] = DS_zps[i].rename({dim: dim+"_c"})

DS_szt  = open_domain_cfg(files=[DOMCFG_szt])
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        DS_szt[i] = DS_szt[i].rename({dim: dim+"_c"})

DS_MEs  = open_domain_cfg(files=[DOMCFG_MEs])
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        DS_MEs[i] = DS_MEs[i].rename({dim: dim+"_c"})

# ==============================================================================

y_str = 'bottom_diff_SPG_average_2014-2018.png'
 
T_zps = Tzpsdir + '/nemo_cn092o_1y_average_2014-2018_grid_T.nc'
T_szt = Tsztdir + '/nemo_cn093o_1y_average_2014-2018_grid_T.nc'
T_MEs = Tmesdir + '/nemo_cn094o_1y_average_2014-2018_grid_T.nc'

# Loading NEMO files
ds_T_zps = open_nemo(DS_zps, files=[T_zps])
ds_T_szt = open_nemo(DS_szt, files=[T_szt])
ds_T_MEs = open_nemo(DS_MEs, files=[T_MEs])

# Extracting only the part of the domain we need

ds_dom_zps = DS_zps.isel(x_c=slice(880,1150),x_f=slice(880,1150),
                         y_c=slice(880,1140),y_f=slice(880,1140))
ds_dom_szt = DS_szt.isel(x_c=slice(880,1150),x_f=slice(880,1150),
                         y_c=slice(880,1140),y_f=slice(880,1140))
ds_dom_MEs = DS_MEs.isel(x_c=slice(880,1150),x_f=slice(880,1150),
                         y_c=slice(880,1140),y_f=slice(880,1140))

ds_T_zps =  ds_T_zps.isel(x_c=slice(880,1150),y_c=slice(880,1140))
ds_T_szt =  ds_T_szt.isel(x_c=slice(880,1150),y_c=slice(880,1140))
ds_T_MEs =  ds_T_MEs.isel(x_c=slice(880,1150),y_c=slice(880,1140))

# Computing model T-levels depth
e3w_3_zps = ds_dom_zps["e3w_0"].values
nk = e3w_3_zps.shape[0]
nj = e3w_3_zps.shape[1]
ni = e3w_3_zps.shape[2]
dep3_zps = np.zeros(shape=(nk,nj,ni))
dep3_zps[0,:,:] = 0.5 * e3w_3_zps[0,:,:]
for k in range(1, nk):
    dep3_zps[k,:,:] = dep3_zps[k-1,:,:] + e3w_3_zps[k,:,:]

e3w_3_szt = ds_dom_szt["e3w_0"].values
dep3_szt = np.zeros(shape=(nk,nj,ni))
dep3_szt[0,:,:] = 0.5 * e3w_3_szt[0,:,:]
for k in range(1, nk):
    dep3_szt[k,:,:] = dep3_szt[k-1,:,:] + e3w_3_szt[k,:,:]

e3w_3_MEs = ds_dom_MEs["e3w_0"].values
dep3_MEs = np.zeros(shape=(nk,nj,ni))
dep3_MEs[0,:,:] = 0.5 * e3w_3_MEs[0,:,:]
for k in range(1, nk):
    dep3_MEs[k,:,:] = dep3_MEs[k-1,:,:] + e3w_3_MEs[k,:,:]

dep4_zps = np.repeat(dep3_zps[np.newaxis, :, :, :], 1, axis=0)
dep4_szt = np.repeat(dep3_szt[np.newaxis, :, :, :], 1, axis=0)
dep4_MEs = np.repeat(dep3_MEs[np.newaxis, :, :, :], 1, axis=0)
ds_T_zps["Tdepth"] = xr.DataArray(dep4_zps,
                                  coords=ds_T_zps["thetao_con"].coords,
                                  dims=ds_T_zps["thetao_con"].dims
                                 )
ds_T_szt["Tdepth"] = xr.DataArray(dep4_szt,
                                  coords=ds_T_szt["thetao_con"].coords,
                                  dims=ds_T_szt["thetao_con"].dims
                                 )
ds_T_MEs["Tdepth"] = xr.DataArray(dep4_MEs,
                                  coords=ds_T_MEs["thetao_con"].coords,
                                  dims=ds_T_MEs["thetao_con"].dims
                                 )

    
bathy = ds_dom_zps["bathymetry"]
lon = ds_dom_zps["glamf"]
lat = ds_dom_zps["gphif"]

# =======================================================================================
# Models' differences

# zps
lev = ds_dom_zps['bottom_level'].load()-1 # because indexes are in Fortran convention
lev = lev.where(lev>0,0) # we removenegative indexes
CT_zps_bot = ds_T_zps['thetao_con'].isel(z_c=lev)[0,:,:].squeeze()
AS_zps_bot = ds_T_zps['so_abs'].isel(z_c=lev)[0,:,:].squeeze()
# Computing potential temperature
PT_zps_bot = gsw.conversions.pt_from_CT(AS_zps_bot.values, CT_zps_bot.values)
# Computing practical salinity
lon2 = ds_dom_zps["glamt"].values
lat2 = ds_dom_zps["gphit"].values
dep_zps_bot = ds_T_zps["Tdepth"].isel(z_c=lev)[0,:,:].squeeze().values
prs = gsw.p_from_z(-dep_zps_bot, lat2)
PS_zps_bot = gsw.SP_from_SA(AS_zps_bot.values, prs, lon2, lat2)

zps_var_T = xr.DataArray(PT_zps_bot,
                         coords=CT_zps_bot.coords,
                         dims=CT_zps_bot.dims
                        )
zps_var_S = xr.DataArray(PS_zps_bot,
                         coords=AS_zps_bot.coords,
                         dims=AS_zps_bot.dims
                        )

# szt
lev = ds_dom_szt['bottom_level'].load()-1 # because indexes are in Fortran convention
lev = lev.where(lev>0,0) # we removenegative indexes
CT_szt_bot = ds_T_szt['thetao_con'].isel(z_c=lev)[0,:,:].squeeze()
AS_szt_bot = ds_T_szt['so_abs'].isel(z_c=lev)[0,:,:].squeeze()
# Computing potential temperature
PT_szt_bot = gsw.conversions.pt_from_CT(AS_szt_bot.values, CT_szt_bot.values)
# Computing practical salinity
lon2 = ds_dom_szt["glamt"].values
lat2 = ds_dom_szt["gphit"].values
dep_szt_bot = ds_T_szt["Tdepth"].isel(z_c=lev)[0,:,:].squeeze().values
prs = gsw.p_from_z(-dep_szt_bot, lat2)
PS_szt_bot = gsw.SP_from_SA(AS_szt_bot.values, prs, lon2, lat2)

szt_var_T = xr.DataArray(PT_szt_bot,
                         coords=CT_szt_bot.coords,
                         dims=CT_szt_bot.dims
                        )
szt_var_S = xr.DataArray(PS_szt_bot,
                         coords=AS_szt_bot.coords,
                         dims=AS_szt_bot.dims
                        )

# MEs
lev = ds_dom_MEs['bottom_level'].load()-1 # because indexes are in Fortran convention
lev = lev.where(lev>0,0) # we removenegative indexes
CT_mes_bot = ds_T_MEs['thetao_con'].isel(z_c=lev)[0,:,:].squeeze()
AS_mes_bot = ds_T_MEs['so_abs'].isel(z_c=lev)[0,:,:].squeeze()
# Computing potential temperature
PT_mes_bot = gsw.conversions.pt_from_CT(AS_mes_bot.values, CT_mes_bot.values)
# Computing practical salinity
lon2 = ds_dom_MEs["glamt"].values
lat2 = ds_dom_MEs["gphit"].values
dep_mes_bot = ds_T_MEs["Tdepth"].isel(z_c=lev)[0,:,:].squeeze().values
prs = gsw.p_from_z(-dep_mes_bot, lat2)
PS_mes_bot = gsw.SP_from_SA(AS_mes_bot.values, prs, lon2, lat2)

mes_var_T = xr.DataArray(PT_mes_bot,
                         coords=CT_mes_bot.coords,
                         dims=CT_mes_bot.dims
                        )
mes_var_S = xr.DataArray(PS_mes_bot,
                         coords=AS_mes_bot.coords,
                         dims=AS_mes_bot.dims
                        )

colmapT = "cubehelix_r"
colmapS = "RdYlBu_r" #nipy_spectral_r" #'PiYG_r'
land = 'gray'
Tmin = -.6
Tmax =  .7
Tstp = 0.1
Smin = -0.06
Smax = 0.07
Sstp = 0.01
ticksT = [-0.6, -0.3, 0., 0.3, 0.6]
ticksS = [-0.06, -0.03, 0., 0.03, 0.06]

# zps - szt

print(' zps-szt T')
Tdiff = zps_var_T - szt_var_T 
cbar_label = ""
fig_name = "zps-szt_ovf_JRA_bot_T_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmapT, Tmin, Tmax, Tstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land=land)

print(' zps-szt S')
Tdiff = zps_var_S - szt_var_S
cbar_label = ""
fig_name = "zps-szt_ovf_JRA_bot_S_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmapS, Smin, Smax, Sstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land=land)

# zps - MEs

print(' zps-MEs T')
Tdiff = zps_var_T - mes_var_T 
cbar_label = ""
fig_name = "zps-mes_ovf_JRA_bot_T_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmapT, Tmin, Tmax, Tstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land=land)# ticks=ticksT)

print(' zps-MEs S')
Tdiff = zps_var_S - mes_var_S
cbar_label = ""
fig_name = "zps-mes_ovf_JRA_bot_S_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmapS, Smin, Smax, Sstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land=land)#, ticks=ticksS)

# szt - MEs

print(' szt-MEs T')
Tdiff = szt_var_T - mes_var_T
cbar_label = ""
fig_name = "szt-mes_ovf_JRA_bot_T_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmapT, Tmin, Tmax, Tstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land=land)

print(' szt-MEs S')
Tdiff = szt_var_S - mes_var_S
cbar_label = ""
fig_name = "szt-mes_ovf_JRA_bot_S_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmapS, Smin, Smax, Sstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land=land)

# colorbar

print(' colorbar T')
Tdiff = szt_var_T - mes_var_T
cbar_label = "T differences @ bot [$^{\circ}$C]"
fig_name = "colbar_diff_T_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmapT, Tmin, Tmax, Tstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land=land, ticks=ticksT)

print(' colorbar S')
Tdiff = szt_var_S - mes_var_S
cbar_label = "S differences @ bot [PSU]"
fig_name = "colbar_diff_S_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmapS, Smin, Smax, Sstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land=land, ticks=ticksS)

