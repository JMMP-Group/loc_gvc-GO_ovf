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
import matplotlib.gridspec as gridspec
import xarray as xr
from xnemogcm import open_domain_cfg, open_nemo
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.feature as feature
import cmocean
import matplotlib.colors as colors

def plot_bot_plume(fig_name, fig_path, lon, lat, var, proj, colmap, vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, bathy=None, cn_lev=None):

    fig = plt.figure(figsize=(20,20), dpi=100)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    ax = fig.add_subplot(spec[:1], projection=proj)
    ax.coastlines()
    ax.add_feature(feature.LAND, color='black',edgecolor='black',zorder=1)

    # Drawing settings
    cmap = colmap #cmocean.cm.deep
    transform = ccrs.PlateCarree()
    if isinstance(vmin,list):
       norm = colors.BoundaryNorm(vmin, cmap.N, clip=True)
       pcol_kwargs = dict(cmap=cmap, norm=norm, transform=transform)
    else:
       pcol_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax, transform=transform)
    #cbar_kwargs = dict(extend=cbar_extend, orientation=cbar_hor)
    #plot_kwargs = dict(color="r", transform=ccrs.PlateCarree())
    land_col = "black"

    # Grid settings
    gl_kwargs = dict()
    gl = ax.gridlines(**gl_kwargs)
    gl.xlines = False
    gl.ylines = False
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 50, 'color': 'k'}
    gl.ylabel_style = {'size': 50, 'color': 'k'}

    # Plotting
    var = var.where(var != 0, -1)
    land = -1. + bathy.where(bathy == 0)
    #print(var)
    pcol = ax.pcolormesh(lon, lat, var, alpha=0.9, **pcol_kwargs)
    if bathy is not None:
       bcon = ax.contour(lon, lat, bathy, levels=cn_lev, colors='k', transform=transform)
       bcon = ax.contour(lon, lat, bathy, levels=[1200.], colors='black', linewidths=7.0, transform=transform)
       bcon = ax.contour(lon, lat, bathy, levels=[2800.], colors='red', linewidths=7.0, transform=transform)
       cmap = plt.cm.get_cmap("viridis").copy()
       bcol = ax.pcolormesh(lon, lat, land, cmap=cmap, vmin=0., vmax=5000, transform=transform)
       bcol.cmap.set_under(land_col)
    #cb = plt.colorbar(pcol, **cbar_kwargs)
    #cb.set_label(label=cbar_label,size=40)
    #cb.ax.tick_params(labelsize=30) 
    ax.set_extent(map_lims)
    print(f"Saving {fig_path}", end=": ")
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    print("done")
    plt.close()

def plot_bathy(fig_name, fig_path, lon, lat, var, proj, colmap, vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims):

    fig = plt.figure(figsize=(20,20), dpi=100)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    ax = fig.add_subplot(spec[:1], projection=proj)
    #ax.coastlines()
    #ax.gridlines()

    # Drawing settings
    cmap = colmap #cmocean.cm.deep
    transform = ccrs.PlateCarree()
    pcol_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax, transform=transform)
    cbar_kwargs = dict(extend=cbar_extend, orientation=cbar_hor)
    #plot_kwargs = dict(color="r", transform=ccrs.PlateCarree())
    land_col = ".15"

    # Grid settings
    gl_kwargs = dict()
    gl = ax.gridlines(**gl_kwargs)
    gl.xlines = False
    gl.ylines = False
    gl.top_labels = True
    gl.right_labels = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 30, 'color': 'k'}
    gl.ylabel_style = {'size': 30, 'color': 'k'}

    # Plotting
    var = var.where(var != 0, -1)
    pcol = ax.pcolormesh(lon, lat, var, **pcol_kwargs)
    pcol.cmap.set_under(land_col)
    bcon = ax.contour(lon.isel(x_f=slice(1, None), y_f=slice(1, None)), lat.isel(x_f=slice(1, None), y_f=slice(1, None)), var, levels=[2800.], colors='red', linewidths=5.0, transform=transform)
    cb = plt.colorbar(pcol, **cbar_kwargs)
    cb.set_label(label=cbar_label,size=40)
    cb.ax.tick_params(labelsize=30)
    ax.set_extent(map_lims)
    print(f"Saving {fig_path}", end=": ")
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    print("done")
    plt.close()

