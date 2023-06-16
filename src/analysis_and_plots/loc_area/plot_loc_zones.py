#!/usr/bin/env python

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.feature as feature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================================
# Input parameters

# 1. INPUT FILES

# Change this to match your local paths set-up
base_dir = "/your_local_path"

sec = 'IB' # 'DS'
loc_file = base_dir + '/loc_gvc-nordic_ovf/models_geometry/loc_area/bathymetry.loc_area.dep2800_novf_sig1_stn9_itr1.nc'

# 2. PLOT
proj = ccrs.Mercator() #ccrs.Robinson()

# 3. SECTIONS
# Iceland-Faroe Ridge 'IB'
sec_lon1 = [ 0.34072625, -3.56557722,-18.569585  ,-26.42872351, -30.314948]
sec_lat1 = [68.26346438, 65.49039963, 60.79252542, 56.24488972,  52.858934]

# Denmark Strait 'DS'
sec_lon2 = [-10.84451672, -25.30818606, -35., -44.081319]
sec_lat2 = [ 71.98049514,  66.73449533,  61.88833838,  56.000932]

# ==============================================================================

# Load localisation file

ds_loc = xr.open_dataset(loc_file)

# Extract only the part of the domain we need 
ds_loc = ds_loc.isel(x=slice(880,1150),y=slice(880,1140))

# Extracting variables
bat = ds_loc.Bathymetry
lat = ds_loc.nav_lat
lon = ds_loc.nav_lon
loc_msk = ds_loc.s2z_msk

#loc_msk = np.ma.array(loc_msk)
loc_s = loc_msk.where(loc_msk==2)
loc_t = loc_msk.where(loc_msk==1)

LLcrnrlon = -46.00 
LLcrnrlat =  50.00 
URcrnrlon =   2.
URcrnrlat =  72.0

map_lims = [LLcrnrlon, URcrnrlon, LLcrnrlat, URcrnrlat]

fig = plt.figure(figsize=(25, 25), dpi=100)
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
ax = fig.add_subplot(spec[:1], projection=proj)
ax.coastlines(linewidth=4, zorder=6)
ax.add_feature(feature.LAND, color='gray',edgecolor='black',zorder=5)

# Drawing settings
transform = ccrs.PlateCarree()

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

cn_lev = [500., 1000., 1500., 2000., 2500., 3000., 3500.]
ax.contour(lon, lat, bat, levels=cn_lev,colors='black',linewidths=1.5, transform=transform, zorder=4)
ax.contour(lon, lat, bat, levels=[2800],colors='forestgreen',linewidths=15., transform=transform, zorder=4)

ax.pcolormesh(lon, lat, loc_s, cmap = 'autumn', transform=transform, zorder=3, alpha=0.6, antialiased=True)
ax.pcolormesh(lon, lat, loc_t, cmap = 'winter_r',transform=transform, zorder=2, alpha=0.9, antialiased=True)

if sec == 'IB':
   ax.plot(sec_lon1, sec_lat1, linewidth=15, color='black', transform=transform, zorder=4)
   ax.plot(sec_lon1, sec_lat1, linewidth=10, color='magenta', transform=transform, zorder=5)
else:
   ax.plot(sec_lon2, sec_lat2, linewidth=15, color='black', transform=transform, zorder=4)
   ax.plot(sec_lon2, sec_lat2, linewidth=10, color='deepskyblue', transform=transform, zorder=5)

ax.set_extent(map_lims)

out_name ='loc_areas_'+sec+'.png'
plt.savefig(out_name,bbox_inches="tight", pad_inches=0.1)
plt.close()

