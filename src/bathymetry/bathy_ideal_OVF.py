#!/usr/bin/env python

from os.path import isfile, basename, splitext
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr

#=======================================================================================
def bresenham_line(x0, x1, y0, y1):
    '''
    point0 = (y0, x0), point1 = (y1, x1)

    It determines the points of an n-dimensional raster that should be 
    selected in order to form a close approximation to a straight line 
    between two points. Taken from the generalised algotihm on
    
    http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    '''
    steep = abs(y1 - y0) > abs(x1 - x0)

    if steep:
       # swap(x0, y0)
       t  = y0
       y0 = x0
       x0 = t
       # swap(x1, y1)    
       t  = y1
       y1 = x1
       x1 = t

    if x0 > x1:
       # swap(x0, x1)
       t  = x1
       x1 = x0
       x0 = t
       # swap(y0, y1)
       t  = y1
       y1 = y0
       y0 = t

    deltax = np.fix(x1 - x0)
    deltay = np.fix(abs(y1 - y0))
    error  = 0.0

    deltaerr = deltay / deltax
    y = y0

    if y0 < y1:
       ystep = 1
    else:
       ystep = -1

    c=0
    pi = np.zeros(shape=[x1-x0+1])
    pj = np.zeros(shape=[x1-x0+1])

    for x in np.arange(x0,x1+1) :
        if steep:
           pi[c]=y
           pj[c]=x
        else:
           pi[c]=x
           pj[c]=y
        error = error + deltaerr
        if error >= 0.5:
           y = y + ystep
           error = error - 1.0
        c += 1

    return pj, pi

#===================================================================================================
def get_poly_line_ij(points_i, points_j):
    '''
    get_poly_line_ij draw rasterised line between vector-points
    
    Description:
      get_poly_line_ij takes a list of points (specified by 
      pairs of indexes i,j) and draws connecting lines between them 
      using the Bresenham line-drawing algorithm.
    
    Syntax:
      line_j, line_i = get_poly_line_ij(points_i, points_j)
    
    Input:
      points_i, points_j: lists of pairs of i, j indexes that define 
                          the line or polyline. The points will be 
                          connected in the order they're given in 
                          these lists. 
    Output:
      line_i, line_j: lists of the i,j coordinates of the points on the
                      rasterised lines. 
    '''
    line_i=[]
    line_j=[]

    line_n=0

    if len(points_i) == 1:
       line_i = points_i
       line_j = points_j
    else:
       for fi in np.arange(len(points_i)-1):
           # start point of the line
           i1 = points_i[fi]
           j1 = points_j[fi]
           # end point of the line
           i2 = points_i[fi+1]
           j2 = points_j[fi+1]
           # 'draw' line from i1,j1 to i2,j2
           pj, pi = bresenham_line(i1,i2,j1,j2)
           if pi[0] != i1 and pj[0] != j1:
              # beginning of the line doesn't match end point, 
              # so we flip both arrays
              pi = np.flipud(pi)
              pj = np.flipud(pj)

           plen = len(pi)

           for PI in np.arange(plen):
               line_n = PI
               if len(line_i) == 0 or line_i[line_n-1] != pi[PI] or line_j[line_n-1] != pj[PI]:
                  line_i.append(int(pi[PI]))
                  line_j.append(int(pj[PI]))

    return line_j, line_i

# ==============================================================================
# 1. Checking for input files
# ==============================================================================
# Change this to match your local paths set-up
base_dir = "/your_local_path"

# Load GO bathymetry
GO_bat = base_dir + "/loc_gvc-nordic_ovf/models_geometry/eORCA025_bathymetry.nc"
ds_bathy = xr.open_dataset(GO_bat).squeeze()
bathy = ds_bathy["Bathymetry"].squeeze()

wrk = bathy.data.copy()

# Creating embayment

i1 = [1040, 1045, 1047, 1042] 
j1 = [1016, 1021, 1019, 1014]
i2 = [1039, 1045, 1048, 1042]
j2 = [1016, 1022, 1019, 1013]

J1, I1 = get_poly_line_ij(i1, j1)
wrk[J1,I1] = 0.
J2, I2 = get_poly_line_ij(i2, j2)
wrk[J2,I2] = 0.

ds_bathy["Bathymetry"].data = wrk
ds_bathy["Bathymetry"].plot()
plt.show()

# -------------------------------------------------------------------------------------   
# Writing the bathy_meter.nc file

out_file = "/your_local_path/loc_gvc-nordic_ovf/models_geometry/eORCA025_bathymetry_ideal.nc"
ds_bathy.to_netcdf(out_file)
