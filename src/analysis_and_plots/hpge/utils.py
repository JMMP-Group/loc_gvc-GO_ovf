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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xarray as xr
from xnemogcm import open_domain_cfg, open_nemo
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.feature as feature
import cmocean

def plot_hpge(fig_name, fig_path, lon, lat, var, proj, colmap, vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, bathy=None, cn_lev=None):

    fig = plt.figure(figsize=(20,20), dpi=100)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    ax = fig.add_subplot(spec[:1], projection=proj)
    ax.coastlines()
    ax.add_feature(feature.LAND, color='black',edgecolor='gray',zorder=1)

    # Drawing settings
    cmap = colmap #cmocean.cm.deep
    transform = ccrs.PlateCarree()
    pcol_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax, transform=transform)
    #cbar_kwargs = dict(extend=cbar_extend, orientation=cbar_hor)
    land_col = ".55"

    # Grid settings
    gl_kwargs = dict()
    gl = ax.gridlines(**gl_kwargs)
    gl.xlines = False
    gl.ylines = False
    gl.top_labels = True
    gl.right_labels = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 50, 'color': 'k'}
    gl.ylabel_style = {'size': 50, 'color': 'k'}

    # Plotting
    pcol = ax.pcolormesh(lon, lat, var, alpha=1., **pcol_kwargs)

    if bathy is not None:
       land = -1. + bathy.where(bathy == 0)
       bcon = ax.contour(lon, lat, bathy, levels=cn_lev, colors='w', transform=transform)
       bcol = ax.pcolormesh(lon, lat, land, **pcol_kwargs)

    
    # Internal colorbar
    if cbar_label != "":
       cbaxes = inset_axes(ax, width="38%", height="3%", loc=2, borderpad=4) 
       cb = plt.colorbar(pcol, cax=cbaxes, ticks=[0., 2.5, 5.],orientation='horizontal', extend=cbar_extend)
       cb.ax.tick_params(color='white', labelcolor='white', labelsize=40, width=4, length=20, direction='in')
       cb.outline.set_edgecolor('white')
       cb.set_label(label=cbar_label,size=45, color='white')

    # External colorbari
    #if cbar_label != "":
    #   cb = plt.colorbar(pcol, ax=ax, ticks=[0., 2.5, 5.], orientation='horizontal', extend=cbar_extend)
    #   cb.ax.tick_params(color='black', labelcolor='black', labelsize=50)
    #   cb.outline.set_edgecolor('black')
    #   cb.set_label(label=cbar_label,size=50, color='black')

    ax.set_extent(map_lims)
    print(f"Saving {fig_path}", end=": ")
    plt.savefig(fig_path+'maximum_'+fig_name, bbox_inches="tight")
    print("done")
    plt.close()

def plot_msk_hpge(fig_name, fig_path, lon, lat, var, proj, colmap, vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, bathy=None, cn_lev=None):

    fig = plt.figure(figsize=(20,20), dpi=100)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    ax = fig.add_subplot(spec[:1], projection=proj)
    ax.coastlines(linewidth=2,zorder=11)
    ax.add_feature(feature.LAND, color='tan',edgecolor='black',zorder=10)

    # Drawing settings
    cmap = colmap #cmocean.cm.deep
    transform = ccrs.PlateCarree()
    pcol_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax, transform=transform,zorder=1)
    #cbar_kwargs = dict(extend=cbar_extend, orientation=cbar_hor)
    land_col = ".55"

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
    pcol = ax.pcolormesh(lon, lat, var, alpha=1., **pcol_kwargs)

    if bathy is not None:
       land = -1. + bathy.where(bathy == 0)
       bcon = ax.contour(lon, lat, bathy, levels=cn_lev, colors='k', transform=transform)
       bcol = ax.pcolormesh(lon, lat, land, **pcol_kwargs)

    # Internal colorbar
    if cbar_label != "":
       cbaxes = inset_axes(ax, width="38%", height="3%", loc=2, borderpad=4)
       cb = plt.colorbar(pcol, cax=cbaxes, ticks=[0., 2.5, 5.],orientation='horizontal', extend=cbar_extend)
       cb.ax.tick_params(color='white', labelcolor='white', labelsize=40, width=4, length=20, direction='in')
       cb.outline.set_edgecolor('white')
       cb.set_label(label=cbar_label,size=45, color='white')

    # External colorbari
    #if cbar_label != "":
    #   cb = plt.colorbar(pcol, ax=ax, ticks=[0., 2.5, 5.], orientation='horizontal', extend=cbar_extend)
    #   cb.ax.tick_params(color='black', labelcolor='black', labelsize=50)
    #   cb.outline.set_edgecolor('black')
    #   cb.set_label(label=cbar_label,size=50, color='black')

    ax.set_extent(map_lims)
    print(f"Saving {fig_path}", end=": ")
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    print("done")
    plt.close()


def plot_env(fig_name, fig_path, lon, lat, var, proj, colmap, vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims):

    fig = plt.figure(figsize=(20,20), dpi=100)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    ax = fig.add_subplot(spec[:1], projection=proj)
    ax.coastlines(linewidth=2)
    #ax.add_feature(feature.LAND, color='white',edgecolor='gray',zorder=10)
    #ax.gridlines()

    # Drawing settings
    cmap = colmap #cmocean.cm.deep
    transform = ccrs.PlateCarree()
    pcol_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax, transform=transform, zorder=1)
    cbar_kwargs = dict(extend=cbar_extend, orientation=cbar_hor)
    #plot_kwargs = dict(color="r", transform=ccrs.PlateCarree())

    # Grid settings
    gl_kwargs = dict()
    gl = ax.gridlines(**gl_kwargs)
    gl.xlines = False
    gl.ylines = False
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 30, 'color': 'k'}
    gl.ylabel_style = {'size': 30, 'color': 'k'}

    # Plotting
    #var = var.where(var != 0, -1)
    #pcol = ax.pcolormesh(lon, lat, var, alpha=1., **pcol_kwargs)
    lev = 20 #np.arange(150.,2800.,100.) 
    pcol = ax.contourf(lon, lat, var, levels=lev, alpha=1., **pcol_kwargs)
    cb = plt.colorbar(pcol, **cbar_kwargs)
    cb.set_label(label=cbar_label,size=50)
    cb.ax.tick_params(labelsize=50)
    ax.set_extent(map_lims)
    print(f"Saving {fig_path}", end=": ")
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    print("done")
    plt.close()

