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

# NOAA data for 2005-2017 are retrieved from:
# https://www.ncei.noaa.gov/data/oceans/ncei/woa/temperature/A5B7/0.25
# https://www.ncei.noaa.gov/data/oceans/ncei/woa/salinity/A5B7/0.25
TOBSdir = base_dir + '/loc_gvc-nordic_ovf/outputs/realistic/obs/NOAA'
SOBSdir = base_dir + '/loc_gvc-nordic_ovf/outputs/realistic/obs/NOAA'
Tzpsdir = base_dir + '/loc_gvc-nordic_ovf/outputs/realistic/zps'
Tsztdir = base_dir + '/loc_gvc-nordic_ovf/outputs/realistic/szt'
Tmesdir = base_dir + '/loc_gvc-nordic_ovf/outputs/realistic/MEs'

# 3. PLOT
lon0 = -61. # -47.
lon1 = -5.  # -2..
lat0 =  46. # 50.
lat1 =  67. # 72.
proj = ccrs.Mercator() #ccrs.Robinson()

fig_path = "./"
cbar_extend = "both"
cn_lev = [500., 1000., 1500, 2000., 3000.]
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

y_str = 'bottom_biases_average_2014-2018.png'
 
# -------------------------
# 1. NOAA WOA observations    
Tfile = TOBSdir + '/woa18_A5B7_t00_04.nc' 
Sfile = SOBSdir + '/woa18_A5B7_s00_04.nc'

# Loading NOAA files
ds_obs_T = xr.open_dataset(Tfile, decode_times=False)
ds_obs_S = xr.open_dataset(Sfile, decode_times=False)

ds_obs_T = ds_obs_T.squeeze()
ds_obs_S = ds_obs_S.squeeze()
   
# Extracting only the part of the domain we need
ds_obs_T = ds_obs_T.isel(lon=slice(300,900), lat=slice(480,790))
ds_obs_S = ds_obs_S.isel(lon=slice(300,900), lat=slice(480,790))

# Computing bottom level
Tan = ds_obs_T.t_an.values
nk = Tan.shape[0]
nj = Tan.shape[1]
ni = Tan.shape[2]
bottom_level = np.zeros((nj,ni), dtype=int)
for j in range(nj):
    for i in range(ni):
        for k in range(nk):
            if np.isnan(Tan[k,j,i]):
               bottom_level[j,i] = k
               break
ds_obs_T['bottom_level'] = xr.DataArray(bottom_level, 
                                        coords=(ds_obs_T.lat, ds_obs_T.lon), 
                                        dims=('lat','lon')
                                       )
ds_obs_S['bottom_level'] = xr.DataArray(bottom_level,
                                        coords=(ds_obs_S.lat, ds_obs_S.lon), 
                                        dims=('lat','lon')
                                       )
lev = ds_obs_T['bottom_level'].load()-1
lev = lev.where(lev>0,0) # we removenegative indexes
depth = ds_obs_T['depth'].load()
daTan = ds_obs_T['t_an']
daSan = ds_obs_S['s_an']
# Computing potential temperature
daSA = gsw.SA_from_SP(daSan, depth, ds_obs_T.lon, ds_obs_T.lat)
daTP = gsw.pt0_from_t(daSA, daTan, depth)
# Extracting values at the bottom
daSan = daSan.assign_coords(depth=range(nk))
daTP = daTP.assign_coords(depth=range(nk))
da_obs_T_bot = daTP.isel(depth=lev)
da_obs_S_bot = daSan.isel(depth=lev)

# -------------------------
# 2. MODELS
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

# A. Interpolating obs to orca025 grid
da_obs_T_bot_int = da_obs_T_bot.interp(lat=ds_dom_zps.gphit, 
                                       lon=ds_dom_zps.glamt,
                                       method='linear')
da_obs_S_bot_int = da_obs_S_bot.interp(lat=ds_dom_zps.gphit, 
                                       lon=ds_dom_zps.glamt,
                                       method='linear')

# =======================================================================================
# B. PLotting NOAA climatology

print(' NOAA T')
Tvar = da_obs_T_bot_int
cbar_label = ""
fig_name = "spg_noaa_ovf_JRA_bot_T_" + y_str

# COLORBAR
cmap    = plt.get_cmap('afmhot_r')
newcmap = truncate_colormap(cmap, 0.0, 0.8)
col_dp  = newcmap(np.linspace(0,1.,128))
newcmap = truncate_colormap(cmocean.cm.ice, 0.2, 1.0)
col_sh  = newcmap(np.linspace(0,1.,128))
col     = list(zip(np.linspace(0,0.5,128),col_sh))
col    += list(zip(np.linspace(0.5,1.0,128),col_dp))
colmapT = colors.LinearSegmentedColormap.from_list('mycmap', col)
vmin = 0 
vmax = 8.
vstp = 0.5
ticks = [0, 2, 4, 6, 8]

plot_bot_plume(fig_name, fig_path, lon, lat, Tvar, proj,
               colmapT, vmin, vmax, vstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy, lon, lat, cn_lev=cn_lev, ticks=ticks)

fig_name = "spg_noaa_colorbar_T"
cbar_label = "T @ bottom [$^{\circ}$C]"
plot_bot_plume(fig_name, fig_path, lon, lat, Tvar, proj,
               colmapT, vmin, vmax, vstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy, lon, lat, cn_lev=cn_lev, ticks=ticks)

print(' NOAA S')
Svar = da_obs_S_bot_int
cbar_label = ""
fig_name = "spg_noaa_ovf_JRA_bot_S_"+y_str

vmin = 34.6
vmax = 35.2
vstp = 0.05
ticks = [34.6, 34.9, 35.2]
cmap = plt.get_cmap('nipy_spectral')
colmapS = truncate_colormap(cmap, 0.1, 1.)

plot_bot_plume(fig_name, fig_path, lon, lat, Svar, proj,
               colmapS, vmin, vmax, vstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, ticks=ticks)

fig_name = "spg_noaa_colorbar_S"
cbar_label = "S @ bottom [PSU]"
plot_bot_plume(fig_name, fig_path, lon, lat, Svar, proj,
               colmapS, vmin, vmax, vstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, ticks=ticks)

# =======================================================================================
# Models' biases

colmap = 'RdBu_r'
Tmin = -1.2
Tmax = 1.3
Tstp = 0.1
Smin = -0.12
Smax = 0.13
Sstp = 0.01
ticksT = [-1.2, -.6, 0., .6, 1.2]
ticksS = [-0.12, -0.06, 0., 0.06, 0.12]

# C. zps - NOAA
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

print(' zps-NOAA T')
Tdiff = zps_var_T - da_obs_T_bot_int
cbar_label = ""
fig_name = "spg_zps-noaa_ovf_JRA_bot_T_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmap, Tmin, Tmax, Tstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy, lon, lat, cn_lev)    

fig_name = "spg_T_diff_colorbar"
cbar_label = "model$-$WOA18 T @ bottom [$^{\circ}$C] "
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmap, Tmin, Tmax, Tstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy, lon, lat, cn_lev, ticksT)

print(' zps-NOAA  S')
Tdiff = zps_var_S - da_obs_S_bot_int
cbar_label = ""
fig_name = "spg_zps-noaa_ovf_JRA_bot_S_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmap, Smin, Smax, Sstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev)

fig_name = "spg_S_diff_colorbar"
cbar_label = "model$-$WOA18 S @ bottom [PSU] "
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmap, Smin, Smax, Sstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, ticks=ticksS)

# =======================================================================================
# D. szt - NOAA
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

print(' szt-NOAA  T')
Tdiff = szt_var_T - da_obs_T_bot_int
cbar_label = ""
fig_name = "spg_szt-noaa_ovf_JRA_bot_T_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmap, Tmin, Tmax, Tstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy, lon, lat, cn_lev, ticks=ticksT)

print(' szt-NOAA S')
Tdiff = szt_var_S - da_obs_S_bot_int
cbar_label = ""
fig_name = "spg_szt-noaa_ovf_JRA_bot_S_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmap, Smin, Smax, Sstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, ticks=ticksS)

# =======================================================================================
# E. MEs - NOAA
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

print(' MEs-NOAA T')
Tdiff = mes_var_T - da_obs_T_bot_int
cbar_label = ""
fig_name = "spg_mes-noaa_ovf_JRA_bot_T_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmap, Tmin, Tmax, Tstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy, lon, lat, cn_lev)

print(' MEs-NOAA S')
Tdiff = mes_var_S - da_obs_S_bot_int
cbar_label = ""
fig_name = "spg_mes-noaa_ovf_JRA_bot_S_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmap, Smin, Smax, Sstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev)


