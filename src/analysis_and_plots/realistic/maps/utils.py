#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import os
from os.path import join, isfile, basename, splitext
import glob
import numpy as np
import nsv
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.path as mpath
import xarray as xr
from xnemogcm import open_domain_cfg, open_nemo
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.mpl.ticker as ctk
import cartopy.feature as feature
import cmocean

def plot_bot_plume(fig_name, fig_path, lon, lat, var, proj, colmap, vmin, vmax, vstp, cbar_extend, cbar_label, cbar_hor, map_lims, bathy=None, lon_bat=None, lat_bat=None, cn_lev=None, ticks=None, ucur=None, vcur=None, land=None):

    fig = plt.figure(figsize=(20,20), dpi=100)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    ax = fig.add_subplot(spec[:1], projection=proj)

    # CONIC SHAPE
    proj.threshold = 1e3

    xlim = [map_lims[0], map_lims[1]]
    ylim = [map_lims[2], map_lims[3]]

    rect = mpath.Path([[xlim[0], ylim[0]],
                       [xlim[1], ylim[0]],
                       [xlim[1], ylim[1]],
                       [xlim[0], ylim[1]],
                       [xlim[0], ylim[0]],
                      ])
    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    rect_in_target = proj_to_data.transform_path(rect)
    ax.set_boundary(rect_in_target)


    ax.coastlines()
    if land is not None:
       ax.add_feature(feature.LAND, color=land,edgecolor=land) #,zorder=10)
    else:
       ax.add_feature(feature.LAND, color='black',edgecolor='black')#,zorder=10)
    #ax.gridlines()

    if isinstance(vstp, list):
       CN_LEV = vstp
    else:
       CN_LEV = np.arange(vmin, vmax, vstp)

    # Drawing settings
    cmap = colmap #cmocean.cm.deep
    transform = ccrs.PlateCarree()
    pcol_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax, extend=cbar_extend, transform=transform)
    #pcol_kwargs = dict(cmap=cmap, extend=cbar_extend, transform=transform)
    #cbar_kwargs = dict(ticks=CN_LEV,orientation=cbar_hor)
    cbar_kwargs = dict(orientation=cbar_hor)

    # Grid settings
    #gl_kwargs = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    #gl = ax.gridlines(**gl_kwargs)
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.xlines = False
    gl.ylines = False
    gl.top_labels = False
    gl.bottom_labels = True 
    gl.right_labels = True
    gl.left_labels = False
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 40, 'color': 'k'}
    gl.ylabel_style = {'size': 40, 'color': 'k'}
    gl.rotate_labels=False
    gl.xlocator=ctk.LongitudeLocator(6)
    gl.ylocator=ctk.LatitudeLocator(4)
    gl.xformatter=ctk.LongitudeFormatter(zero_direction_label=False)
    gl.yformatter=ctk.LatitudeFormatter()

    # Plotting
    var = var.where(var != 0, np.nan)
    #var = var.where(np.isfinite(var), -1)
    pcol = ax.contourf(lon, lat, var, levels=CN_LEV, **pcol_kwargs)
    #pcol.cmap.set_over("lightgray")
    #pcol.cmap.set_under("lightcyan")
    #pcol.set_clim(CN_LEV[0], CN_LEV[-1])

    if bathy is not None:
       if lon_bat is not None and lat_bat is not None:
          bcon = ax.contour(lon_bat, 
                            lat_bat, 
                            bathy, 
                            levels=cn_lev, 
                            colors='orange', 
                            linewidths=3.0, 
                            transform=transform)
       else:
          bcon = ax.contour(lon, lat,
                            bathy, 
                            levels=cn_lev, 
                            colors='orange', 
                            linewidths=3.0,
                            transform=transform)

    if ucur is not None and vcur is not None:
    
       stp = 2
       ucur = ucur.where(np.isfinite(var))
       vcur = vcur.where(np.isfinite(var))
       speed = np.sqrt(ucur**2 + vcur**2)
       speed = speed.where(np.isfinite(var))
       ucur = ucur.where(speed>0.05)
       vcur = vcur.where(speed>0.05)

       QV = ax.quiver(lon.values[::stp,::stp], lat.values[::stp,::stp], \
                      ucur.values[::stp,::stp], vcur.values[::stp,::stp], \
                      transform=transform, \
                      color='k', \
                      units='xy',\
                      angles='xy', \
                      #scale=1.,
                      scale=.5,
                      scale_units='inches')#,
                      #linewidth=0.01, 
                      #edgecolors='k')
       qk = plt.quiverkey(QV, 0.18, 0.75, 1, "1 Sv", 
                          labelpos = "S", 
                          coordinates='figure', 
                          fontproperties={'size': 40, 'weight': 'bold'},
                          color='w', 
                          labelcolor='w')
       qk.text.set_zorder(12)
       
    # osnap EAST
    ds = eval("nsv.Standardizer().osnap")
    ds = ds.isel(station=slice(80, None))
    lon = ds.longitude
    lat = ds.latitude
    lon_IS = lon.where(lon<=-31.2) # Irminger Sea
    lat_IS = lat.where(lon<=-31.2)
    lon_IB = lon.where(np.logical_and(lon>-31.2,lon<=-13.2)) # Icelandic Basin
    lat_IB = lat.where(np.logical_and(lon>-31.2,lon<=-13.2))

    ax.plot(lon_IS.values, 
            lat_IS.values, 
            color='k', 
            linewidth=15, 
            transform=ccrs.PlateCarree())
    ax.plot(lon_IS.values, 
            lat_IS.values, 
            color='gold', 
            label='IS', 
            linewidth=10, 
            transform=ccrs.PlateCarree())

    ax.plot(lon_IB.values, 
            lat_IB.values, 
            color='k', 
            linewidth=15, 
            transform=ccrs.PlateCarree())
    ax.plot(lon_IB.values, 
            lat_IB.values, 
            color='limegreen', 
            label='IS', 
            linewidth=10, 
            transform=ccrs.PlateCarree())

    # External colorbar
    if cbar_label != "":    
       if ticks is not None:
          cb = plt.colorbar(pcol, ax=ax, ticks=ticks, orientation=cbar_hor, extend=cbar_extend, drawedges=True)
       else:
          cb = plt.colorbar(pcol, ax=ax, orientation=cbar_hor, extend=cbar_extend, drawedges=True)
       cb.ax.tick_params(color='black', labelcolor='black', labelsize=50, width=4, length=20, direction='in')
       #cb.outline.set_edgecolor('white')
       cb.set_label(label=cbar_label, size=50, color='black')
       cb.outline.set_color('black')
       cb.outline.set_linewidth(4)
       cb.dividers.set_color('black')
       cb.dividers.set_linewidth(4)

    #cb = plt.colorbar(pcol, **cbar_kwargs)
    #cb.set_label(label=cbar_label,size=40)
    #cb.ax.tick_params(labelsize=40)

    # Internal colorbar
    #if cbar_label != "":
    #   if cbar_hor == 'horizontal':
    #      cbaxes = inset_axes(ax, width="38%", height="3%", loc=2, borderpad=4)
    #   else:
    #      cbaxes = inset_axes(ax, width="3%", height="48%", loc=2, borderpad=2) 
    #   if ticks is not None:
    #      cb = plt.colorbar(pcol, cax=cbaxes, ticks=ticks, orientation=cbar_hor, extend=cbar_extend)
    #   else:
    #      cb = plt.colorbar(pcol, cax=cbaxes, orientation=cbar_hor, extend=cbar_extend)
    #   cb.ax.tick_params(color='white', labelcolor='white', labelsize=35, width=4, length=20, direction='in')
    #   cb.outline.set_edgecolor('white')
    #   cb.set_label(label=cbar_label, size=40, color='white', rotation=0, y=1.05, labelpad=160)

    ax.set_extent(map_lims, crs=ccrs.PlateCarree())

    print(f"Saving {fig_path}", end=": ")
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    print("done")
    plt.close()

def e3_to_dep(e3W, e3T):

    gdepT = xr.full_like(e3T, None, dtype=np.double).rename('gdepT')
    gdepW = xr.full_like(e3W, None, dtype=np.double).rename('gdepW')

    gdepW[{"z_f":0}] = 0.0
    gdepT[{"z_c":0}] = 0.5 * e3W[{"z_f":0}]
    for k in range(1, e3W.sizes["z_f"]):
        gdepW[{"z_f":k}] = gdepW[{"z_f":k-1}] + e3T[{"z_c":k-1}]
        gdepT[{"z_c":k}] = gdepT[{"z_c":k-1}] + e3W[{"z_f":k}]

    return tuple([gdepW, gdepT])

def compute_masks(ds_domain, merge=False):
    """
    Compute masks from domain_cfg Dataset.
    If merge=True, merge with the input dataset.
    Parameters
    ----------
    ds_domain: xr.Dataset
        domain_cfg datatset
    add: bool
        if True, merge with ds_domain
    Returns
    -------
    ds_mask: xr.Dataset
        dataset with masks
    """

    # Extract variables
    k = ds_domain["z_c"] + 1
    top_level = ds_domain["top_level"]
    bottom_level = ds_domain["bottom_level"]

    # Page 27 NEMO book.
    # I think there's a typo though.
    # It should be:
    #                  | 0 if k < top_level(i, j)
    # tmask(i, j, k) = | 1 if top_level(i, j) ≤ k ≤ bottom_level(i, j)
    #                  | 0 if k > bottom_level(i, j)
    tmask = xr.where(np.logical_or(k < top_level, k > bottom_level), 0, np.nan)
    tmask = xr.where(np.logical_and(bottom_level >= k, top_level <= k), 1, tmask)
    tmask = tmask.rename("tmask")

    tmask = tmask.transpose("z_c","y_c","x_c")

    # Need to shift and replace last row/colum with tmask
    # umask(i, j, k) = tmask(i, j, k) ∗ tmask(i + 1, j, k)
    umask = tmask.rolling(x_c=2).prod().shift(x_c=-1)
    umask = umask.where(umask.notnull(), tmask)
    umask = umask.rename("umask")

    # vmask(i, j, k) = tmask(i, j, k) ∗ tmask(i, j + 1, k)
    vmask = tmask.rolling(y_c=2).prod().shift(y_c=-1)
    vmask = vmask.where(vmask.notnull(), tmask)
    vmask = vmask.rename("vmask")

    # Return
    masks = xr.merge([tmask, umask, vmask])
    if merge:
        return xr.merge([ds_domain, masks])
    else:
        return masks

