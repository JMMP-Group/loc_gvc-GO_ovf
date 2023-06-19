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
from utils import plot_bot_plume
import gsw

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

YYYY = 2015
y1 = str(YYYY)
y2 = str(YYYY)

models = ["zps","szt","MEs"]
runsid = ["cn092", "cn093", "cn094"]

# Change this to match your local paths set-up
base_dir = "/your_local_path"

DOMCFG = [base_dir + '/models_geometry/dom_cfg/realistic/zps/domain_cfg_zps.nc',
          base_dir + '/models_geometry/dom_cfg/realistic/szt/domain_cfg_szt.nc',
          base_dir + '/models_geometry/dom_cfg/realistic/mes/domain_cfg_mes.nc'
         ]

# 3. PLOT
lon0 = -45. # -60.
lon1 = -5. #0.
lat0 =  51. #45.
lat1 =  67. # 72.
proj = ccrs.Mercator() #ccrs.Robinson()

# COLORBAR
cmap    = plt.get_cmap('afmhot_r')
newcmap = truncate_colormap(cmap, 0.0, 0.8)
col_dp  = newcmap(np.linspace(0,1.,128))
newcmap = truncate_colormap(cmocean.cm.ice, 0.1, 1.0)
col_sh  = cmocean.cm.ice(np.linspace(0,1.,128))
col     = list(zip(np.linspace(0,0.5,128),col_sh))
col    += list(zip(np.linspace(0.5,1.0,128),col_dp))
CMAP   = colors.LinearSegmentedColormap.from_list('mycmap', col)
norm = TwoInnerPointsNormalize(vmin=0, vmax=0.5, low=0.1, up=0.2)

# ==============================================================================

outdir= base_dir + '/realistic/'

for m in range(len(models)):

    T = outdir + models[m] + '/nemo_' + runsid[m] + 'o_1y_average_2014-2018_grid_T.nc'
    U = outdir + models[m] + '/nemo_' + runsid[m] + 'o_1y_average_2014-2018_grid_U.nc'
    V = outdir + models[m] + '/nemo_' + runsid[m] + 'o_1y_average_2014-2018_grid_V.nc'

    # Loading domain geometry
    ds_dom = open_domain_cfg(files=[DOMCFG[m]])
    for i in ['bathymetry','bathy_meter']:
        for dim in ['x','y']:
            ds_dom[i] = ds_dom[i].rename({dim: dim+"_c"})

    # Loading NEMO files
    ds_T = open_nemo(ds_dom, files=[T])
    ds_U = open_nemo(ds_dom, files=[U])
    ds_V = open_nemo(ds_dom, files=[V])

    # Extracting only the part of the domain we need

    ds_dom = ds_dom.isel(x_c=slice(880,1150),x_f=slice(880,1150),
                         y_c=slice(880,1140),y_f=slice(880,1140))

    bathy = ds_dom["bathymetry"]
    lon = ds_dom["glamf"]
    lat = ds_dom["gphif"]
    e1t = ds_dom["e1t"]
    e2t = ds_dom["e2t"]
    e3t = ds_dom["e3t_0"]

    ds_T = ds_T.isel(x_c=slice(880,1150),y_c=slice(880,1140))
    ds_U = ds_U.isel(x_f=slice(880,1150),y_c=slice(880,1140))
    ds_V = ds_V.isel(x_c=slice(880,1150),y_f=slice(880,1140))

    # Regridding U and V on T points

    Uc = ds_U['uo'].squeeze()
    Vc = ds_V['vo'].squeeze()
    u = Uc.rolling({'x_f':2}).mean().fillna(0.)
    v = Vc.rolling({'y_f':2}).mean().fillna(0.)
    u = u.rename({'x_f':'x_c'})
    v = v.rename({'y_f':'y_c'})
    u = u.assign_coords({"x_c": e1t['x_c'].data})
    v = v.assign_coords({"y_c": e1t['y_c'].data})

    # Computing potential density anomaly
    rho = gsw.density.sigma0(ds_T.so_abs.squeeze(), ds_T.thetao_con.squeeze())

    # Only grid cells where sigma_theta > 27.84 
    ovf_rho = 27.84 # 27.80
    u = u.where(rho > ovf_rho)
    v = v.where(rho > ovf_rho)
    e1t = e1t.where(rho > ovf_rho)
    e2t = e2t.where(rho > ovf_rho)
    e3t = e3t.where(rho > ovf_rho).squeeze()

    uflux = (u * e2t * e3t).sum(dim='z_c', skipna=True) * 1e-6 # in Sv
    vflux = (v * e1t * e3t).sum(dim='z_c', skipna=True) * 1e-6 # in Sv

    print(np.nanmax(uflux), np.nanmax(vflux))

    ovf_thickness = e3t.sum('z_c', skipna=True) * 1.e-3 # in km

    # PLOT =============================================================================

    fig_path = "./"
    colmap = "rainbow" #CMAP
    cbar_extend = "max"
    cn_lev = [500., 1000., 1500., 2000., 3000.]
    cbar_hor = 'vertical'
    map_lims = [lon0, lon1, lat0, lat1]
    y_str = '2014-2018.png'

    vmin = 0.
    vmax = 2.2
    vstp = 0.2
    cbar_label = ""
    if m == 3: cbar_label = "Overflow layer thickness [$km$]"
    vcor = models[m]

    fig_name = "ovf_transport_" + vcor + "_" + str(ovf_rho) + "_" + y_str
    if m == 3: fig_name = "colorbar_transport.png"
    var = ovf_thickness
  
    plot_bot_plume(fig_name, fig_path, lon, lat, var, proj,
                   colmap, vmin, vmax, vstp, cbar_extend, cbar_label, 
                   cbar_hor, map_lims, bathy, cn_lev=cn_lev, ucur=uflux.squeeze(), vcur=vflux.squeeze())

