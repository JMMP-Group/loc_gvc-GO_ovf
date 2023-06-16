#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|

import argparse
import re
import gsw
import numpy as np
import xarray as xr
from xnemogcm import open_domain_cfg, open_nemo
# add line for skimage
# add geopy for haversine dist
from scipy.interpolate import interp1d

# ============================================================================================
def getArgs():
    """
    Checks input options and handles exceptions 
    """
    sections = ['latrabjarg_climatology','kogur','ovide','osnap-seasonal','osnap-annual',
                'ho2000','eel','pos503-5','m82_1-1','m82_1-2','m82_1-3','m82_1-4','m82_1-5',
                'm82_1-6','m82_1-7','m82_1-8','m82_1-9','kn203_2-A','kn203_2-B','kn203_2-C',
                'kn203_2-D','kn203_2-E']
    vcoord = ['zps', 'mes', 'szt']

    parser = argparse.ArgumentParser(description='''
                                                 Code to plot T or S sections 
                                                 from observations or model outputs
                                                 ''')
    group = parser.add_mutually_exclusive_group()
    parser.add_argument(
           'sec',
           type=str,
           choices=sections,
           nargs=1,
           help='Section that you want to plot. Allowed values are: '+ \
                ', '.join(sections),
           metavar='Sec'
    )
    parser.add_argument(
           '-locmsk',
           metavar='LOCMSK_PATH',
           type=str,
           nargs=1,
           help='Path of the file including the localisation mask (i.e., bathy_meter.nc).'
    )
    parser.add_argument(
           '-domcfg',
           metavar='DOMAIN_CFG_PATH',
           type=str,
           nargs=1,
           help='Path of the domain_cfg.nc file you want '+ \
           'to use for your plot.'
    )
    group.add_argument(
          '-zps',
          metavar='FILE_PATH',
          type=str,
          nargs=1,
          help='''
               Specify this flag if you want to plot zps model data.
               You need to provide the path of model output you want to plot. 
               '''
    )
    group.add_argument(
          '-mes',
          metavar='FILE_PATH',
          type=str,
          nargs=1,
          help='''
               Specify this flag if you want to plot MEs model data.
               You need to provide the path of model output you want to plot.
               '''
    )
    group.add_argument(
          '-szt',
          metavar='FILE_PATH',
          type=str,
          nargs=1,
          help='''
               Specify this flag if you want to plot szt model data.
               You need to provide the path of model output you want to plot.
               '''
    )
    group.add_argument(
          '-obs',
          action='store_true',
          help='Specify this flag if you want to plot observational data.'
    )

    args = parser.parse_args()

    model = args.zps or args.mes
    if model and args.domcfg is None:
       parser.error("[-zps | -mes | -szt] require [-domcfg DOMAIN_CFG_PATH].")
    if (args.mes and args.locmsk is None) or (args.szt and args.locmsk is None):
       parser.error("[-mes] require [-locmsk LOCMSK_PATH].")
    if args.obs and args.domcfg is not None:
       parser.error("[-obs] does not require [-domcfg DOMAIN_CFG_PATH].")
    if args.obs and args.locmsk is not None:
       parser.error("[-obs] does not require [-locmsk LOCMSK_PATH].") 

    return args

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


def compute_levels(ds, merge=True):
    e3t = ds.e3t_0.values
    e3w = ds.e3w_0.values 
    nk = e3t.shape[0]
    nj = e3t.shape[1]
    ni = e3t.shape[2]
    tdep3 = np.zeros(shape=(nk,nj,ni))
    wdep3 = np.zeros(shape=(nk,nj,ni))
    wdep3[0,:,:] = 0.
    tdep3[0,:,:] = 0.5 * e3w[0,:,:]
    for k in range(1, nk):
        wdep3[k,:,:] = wdep3[k-1,:,:] + e3t[k-1,:,:]
        tdep3[k,:,:] = tdep3[k-1,:,:] + e3w[k,:,:]

    da_t = xr.DataArray(data=tdep3, dims=ds.e3t_0.dims)
    da_w = xr.DataArray(data=wdep3, dims=ds.e3w_0.dims)
    depths = xr.merge([da_t.rename("gdept_0"), da_w.rename("gdepw_0")])

    if merge:
        return xr.merge([ds, depths])
    else:
        return depths

def compute_potential_sigma(ds):
    # Absolute Salinity
    press = gsw.p_from_z(-ds.z, ds.latitude)
    abs_s = gsw.SA_from_SP(ds.practical_salinity, 
                           press, 
                           ds.longitude,  
                           ds.latitude
    )
    # Conservative Temperature
    con_t = gsw.CT_from_pt(abs_s, 
                           ds.potential_temperature
    )
    # Potential density anomaly
    ds['sigma_theta'] = gsw.density.sigma0(abs_s, con_t)

    return ds


def compute_topo(ds, ds_T):
    var = [v.lower() for v in list(ds.keys())]
    r = re.compile(".*floor_depth")
    topo = list(filter(r.match, var))

    if not topo:
       msk2 = ds_T.mask.isel(t=0,y=0).values
       dep2 = ds_T.depth.isel(t=0,y=0).values
       ni = len(ds_T.mask.x)
       nk = len(ds_T.mask.z)
       nj = 5
       bat1 = np.ones(shape=(ni,))*np.nanmax(dep2)
       for i in range(ni):
           for k in range(nk):
               if msk2[k,i] == 0:
                  bat1[i] = dep2[k,i]
                  break
       bat2 = np.repeat(bat1[np.newaxis, :], nj, axis=0)
       ds_T["sea_floor_depth_below_geoid"] = (("y","x"), bat2)
    else:
       da_bathy = ds.sea_floor_depth_below_geoid
       da_bathy = da_bathy.drop(["latitude", "longitude", "distance"])
       da_bathy = da_bathy.rename({"station":"x"})
       da_bathy = da_bathy.expand_dims({"y": 5})
       da_bathy = da_bathy.transpose("y","x")
       ds_T = xr.merge([ds_T, da_bathy])

    return ds_T


def extract_obs(ds):
    """
    Create temporal averages if needed and reshape 
    the arrays we need for the plot in a 3D dataset
    suitable for the plotting routine.
    """
    
    # Computing temporal averages
    for sec in ["kogur", "osnap-seasonal"]:
        if sec.upper() in ds.description:
           seasonal = True
           break
    else:
        seasonal = False
    #multiyear = True if "eel".upper() in ds.description else False
    for sec in ["osnap-annual","eel"]:
        if sec.upper() in ds.description:
           multiyear = True
           break
    else:
        multiyear = False
    #seasonal = True if "kogur".upper() in ds.description else False

    if seasonal:
       # Computing SEASONAL AVERAGES
       # 1) Create monthly dates to be used
       #    as coordinates
       ds = xr.decode_cf(ds) 
       dates = xr.cftime_range(
            start=str(ds.time.dt.date[0].values),
            periods=len(ds.groupby("time.month")),
            freq="MS",
            calendar="noleap",
       )
       print(ds)
       # 2) Compute monthly averages
       ds = ds.groupby("time.month").mean(keep_attrs=True)
       ds = ds.assign_coords(month=dates)
       ds = ds.rename({"month": "time"}) 
       # 3) Make a DataArray with the number of 
       #    days in each month, size = len(months)
       month_length = ds.time.dt.days_in_month
       # 4) Calculate the weights by 
       #    grouping by 'time.season'
       weights = month_length.groupby("time.season") / month_length.groupby("time.season").sum(keep_attrs=True)
       # Test that the sum of the weights for each season is 1.0
       #np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))
       # Calculate the weighted average
       description = ds.description
       ds = (ds * weights).groupby("time.season").sum(dim="time",keep_attrs=True)
       ds.attrs['description'] = description

    if multiyear:
       ds = ds.mean("time", skipna=True, keep_attrs=True)

    # Renaming salinity
    var = [v.lower() for v in list(ds.keys())]
    r = re.compile(".*salinity")
    sal = list(filter(r.match, var))
    if not sal:
       ValueError('No salinity in your data!!')
    else:
       ds = ds.rename_vars({sal[0]:"practical_salinity"})

    # Extracting T and S and renaming coordinates:
    ds_T = xr.merge([ds.potential_temperature, ds.practical_salinity])
    ds_T = ds_T.rename_dims({"station":"x","depth": "z"})
    ds_T = ds_T.reset_coords(["latitude", "longitude", "distance"])
    ds_T["depth"] = ds_T.depth.expand_dims({"x":ds_T.x})
    ds_T = ds_T.reset_coords(["depth"])
    ds_T["longitude"] = ds_T.longitude.expand_dims({"z":ds_T.z})
    ds_T["latitude"] = ds_T.latitude.expand_dims({"z":ds_T.z})

    # Managing time dimension
    if seasonal:
        ds_T = ds_T.rename({"season":"t"})
        ds_T["depth"] = ds_T.depth.expand_dims({"t":ds_T.t})
        ds_T["longitude"] = ds_T.longitude.expand_dims({"t":ds_T.t})
        ds_T["latitude"] = ds_T.latitude.expand_dims({"t":ds_T.t})
    else:
        ds_T = ds_T.expand_dims({"t":[0]})

    # New dimension:
    ds_T = ds_T.expand_dims({"y": 5})

    # Reordering dimensions:
    ds_T = ds_T.transpose("t","z","y","x")

    # Computing land-sea mask
    tracer = (
          ds_T.practical_salinity.isel(t=0) 
          if seasonal 
          else ds_T.practical_salinity
    )
    if "pos503".upper() in ds.description:
       mask = (
           xr.where(np.logical_and(np.isnan(tracer),ds_T["depth"]>100.), 0, 1)
       )
       ds = ds.drop("sea_floor_depth_below_geoid")
    else:
       mask = (
           xr.where(tracer==0, 0, 1)
           if np.isnan(tracer).sum() == 0 
           else xr.where(np.isnan(tracer), 0, 1)
       )
    if seasonal: mask = mask.expand_dims({"t": 1})
    ds_T["mask"] = mask

    # Compute potential density 
    ds_T = compute_potential_sigma(ds_T)  

    # If needed, compute bathymetry
    ds_T = compute_topo(ds, ds_T)

    return ds_T    

   
def extract_model(domcfg, File, xmin, xmax, ymin, ymax):
    
    # Loading domain geometry
    ds_dom  = open_domain_cfg(files=[domcfg])
    for i in ['bathymetry','bathy_meter']:
        if i in ds_dom:
           for dim in ['x','y']:
               ds_dom[i] = ds_dom[i].rename({dim: dim+"_c"})

    # Loading NEMO files
    ds_T = open_nemo(ds_dom, files=[File])

    # Extracting only the portion of 
    # the domain we are interested in
    ds_dom = ds_dom.isel(
                 x_c=slice(xmin, xmax),
                 x_f=slice(xmin, xmax),
                 y_c=slice(ymin, ymax),
                 y_f=slice(ymin, ymax)
    )
    ds_T =  ds_T.isel(x_c=slice(xmin, xmax),y_c=slice(ymin, ymax))

    # Computing model levels depth if needed
    if not ("gdept_0" in ds_dom and "gdepw_0" in ds_dom):
       ds_dom = compute_levels(ds_dom, merge=True)

    # Computing masks
    ds_dom = compute_masks(ds_dom, merge=True)
    ds_dom = ds_dom.reset_coords(["glamt", "gphit", 
                                  "gdept_0", "gdepw_0"
                                 ])

    da_glamt = ds_dom.glamt
    da_gphit = ds_dom.gphit
    da_glamt = da_glamt.expand_dims({"z_c": ds_dom.z_c})
    da_gphit = da_gphit.expand_dims({"z_c": ds_dom.z_c})
    ds_dom = xr.merge([
                  da_glamt,
                  da_gphit,     
                  ds_dom.gdept_0,
                  ds_dom.gdepw_0,     
                  ds_dom.tmask,
                  ds_dom.bathymetry
    ])

    # Computing potential temperature
    da_PT = gsw.conversions.pt_from_CT(ds_T.so_abs, ds_T.thetao_con)

    # Computing practical salinity
    da_PR = gsw.p_from_z(-ds_dom.gdept_0, ds_dom.gphit)
    da_PS = gsw.SP_from_SA(ds_T.so_abs, da_PR, ds_dom.glamt, ds_dom.gphit)

    # Computing potential density anomaly
    da_RH = gsw.density.sigma0(ds_T.so_abs, ds_T.thetao_con)

    return xr.merge([ds_dom, 
                     da_PT.rename("potential_temperature"), 
                     da_PS.rename("practical_salinity"), 
                     da_RH.rename("sigma_theta")])


# =====================================================================================================

def hvrsn_dst(lon1, lat1, lon2, lat2):
    '''
    This function calculates the great-circle distance in meters between 
    point1 (lon1,lat1) and point2 (lon2,lat2) using the Haversine formula 
    on a spherical earth of radius 6378.137km. 

    The great-circle distance is the shortest distance over the earth's surface.
    ( see http://www.movable-type.co.uk/scripts/latlong.html)

    --------------------------------------------------------------------------------

    If lon2 and lat2 are 2D matrixes, then dist will be a 2D matrix of distances 
    between all the points in the 2D field and point(lon1,lat1).

    If lon1, lat1, lon2 and lat2 are vectors of size N dist wil be a vector of
    size N of distances between each pair of points (lon1(i),lat1(i)) and 
    (lon2(i),lat2(i)), with 0 => i > N .
    '''

    deg2rad = np.pi / 180.
#    ER = 6378.137 * 1000. # Earth Radius in meters
    ER = 6372.8 * 1000. # Earth Radius in meters

    dlon = np.multiply(deg2rad, (lon2 - lon1))
    dlat = np.multiply(deg2rad, (lat2 - lat1))

    lat1 = np.multiply(deg2rad, lat1)
    lat2 = np.multiply(deg2rad, lat2)

    # Computing the square of half the chord length between the points:
    a = np.power(np.sin(np.divide(dlat, 2.)),2) + \
        np.multiply(np.multiply(np.cos(lat1),np.cos(lat2)),np.power(np.sin(np.divide(dlon, 2.)),2))

    # Computing the angular distance in radians between the points
    angle = np.multiply(2., np.arctan2(np.sqrt(a), np.sqrt(1. -a)))

    # Computing the distance 
    dist = np.multiply(ER, angle)

    return dist

# =====================================================================================================

def get_ij_from_lon_lat(lon, lat, glamt, gphit):
    '''
    get_ij_from_lon_lat find closest model grid point i/j from given lat/lon

    Syntax:
      [i, j = get_ij_from_lon_lat(lon, lat, glamt, gphit)
   
    Description:
      returns the i,j model grid position which is closest to the given
      lat/lon point. The model grid is given by 2D lon and lat matrix glamt
      and gphit.
    '''

    dist = hvrsn_dst(lon, lat, glamt, gphit)

    min_dist = np.amin(dist)

    find_min = np.where(dist == min_dist)
    sort_j = np.argsort(find_min[1])

    j_indx = find_min[0][sort_j]
    i_indx = find_min[1][sort_j]

    return j_indx[0], i_indx[0]

# =====================================================================================================

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

# =====================================================================================================

def get_poly_line_ij(points_i, points_j):
    '''
    get_poly_line_ij draw rasterised line between vector-points
    
    Description:
    get_poly_line_ij takes a list of points (specified by 
    pairs of indexes i,j) and draws connecting lines between them 
    using the Bresenham line-drawing algorithm.
    
    Syntax:
    line_i, line_j = get_poly_line_ij(points_i, points_i)
    
    Input:
    points_i, points_j: vectors of equal length of pairs of i, j
                        coordinates that define the line or polyline. The
                        points will be connected in the order they're given
                        in these vectors. 
    Output:
    line_i, line_j: vectors of the same length as the points-vectors
                    giving the i,j coordinates of the points on the
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
           # start point of line
           i1 = points_i[fi]
           j1 = points_j[fi]
           # end point of line
           i2 = points_i[fi+1]
           j2 = points_j[fi+1]
           # 'draw' line from i1,j1 to i2,j2
           pj, pi = bresenham_line(i1,i2,j1,j2)
           if pi[0] != i1 or pj[0] != j1:
              # beginning of line doesn't match end point, 
              # so we flip both vectors
              pi = np.flipud(pi)
              pj = np.flipud(pj)

           plen = len(pi)

           for PI in np.arange(plen):
               line_n = PI
               if len(line_i) == 0 or line_i[line_n-1] != pi[PI] or line_j[line_n-1] != pj[PI]:
                  line_i.append(int(pi[PI]))
                  line_j.append(int(pj[PI]))


    return line_j, line_i

# =====================================================================================================

def create_model_bathy_sec(vlevel, msk2, xcoord2, tdep2, wdep2, max_dep=[]):
    ''' 
    This function returns the vertexs which 
    define the model bathymetry. These vertexes 
    can be used to create a patch of the model 
    bathymetry.

    msk2: 2D matrix, slice of tmask.
          N.B this matrix has to be modified
              so that
                       land  = 1.
                       ocean = np.nan
    xcoord2: 2D matrix, x coordinates of the slice
    tdep2:   2D matrix, slice of gdept_0
    wdep2:   2D matrix, slice of gdepw_0
    vlevel:  type of model vertical geometry
    '''
    nz = msk2.shape[0]
    nx = msk2.shape[1]

    z_indx = []

    for i in np.arange(nx):
        for k in np.arange(1,nz):
            if k == 1 and msk2[k-1,i] == 1:
               #print tdep2[k-1,i], wdep2[k-1,i]
               #print i, k-1
               z_indx.append(k-1)
               break
            else:
               if msk2[k,i] == 1 and np.isnan(msk2[k-1,i]):
                  #print tdep2[k,i], wdep2[k,i] 
                  #print i, k
                  z_indx.append(k)
                  break
    vert_x = []
    vert_z = []

    if vlevel == "Z_fs" or vlevel == "MES" or vlevel == "SZT":
    # For MES this is not very accurate: 
    # if levels are very wide, the plotting 
    # is very inaccurate.
       for i in range(nx):
           if i == 0:
              left_z  = wdep2[z_indx[i],i]
              right_z = wdep2[z_indx[i],i+1]
              left_x  = xcoord2[z_indx[i],i]
              right_x = xcoord2[z_indx[i],i] + 0.5*(xcoord2[z_indx[i],i+1] - xcoord2[z_indx[i],i])
           elif i > 0 and i < nx-1:
              left_z  = wdep2[z_indx[i],i]
              right_z = wdep2[z_indx[i],i+1]
              left_x  = xcoord2[z_indx[i],i] - 0.5*(xcoord2[z_indx[i],i] - xcoord2[z_indx[i],i-1])
              right_x = xcoord2[z_indx[i],i] + 0.5*(xcoord2[z_indx[i],i+1] - xcoord2[z_indx[i],i])
           else:
              left_z  = wdep2[z_indx[i],i]
              right_z = wdep2[z_indx[i],i]
              left_x   = xcoord2[z_indx[i],i] - 0.5*(xcoord2[z_indx[i],i] - xcoord2[z_indx[i],i-1])
              right_x  = xcoord2[z_indx[i],i]
           vert_z.append(left_z)
           vert_z.append(right_z)
           vert_x.append(left_x)
           vert_x.append(right_x)
    else:
       for i in range(nx):
           vert_z.append(wdep2[z_indx[i],i])
           vert_z.append(wdep2[z_indx[i],i])
           if i == 0:
              left_x  = xcoord2[z_indx[i],i]
              right_x = xcoord2[z_indx[i],i] + 0.5*(xcoord2[z_indx[i],i+1] - xcoord2[z_indx[i],i])
           elif i > 0 and i < nx-1:
              left_x  = xcoord2[z_indx[i],i] - 0.5*(xcoord2[z_indx[i],i] - xcoord2[z_indx[i],i-1])
              right_x = xcoord2[z_indx[i],i] + 0.5*(xcoord2[z_indx[i],i+1] - xcoord2[z_indx[i],i])
           else:
              left_x   = xcoord2[z_indx[i],i] - 0.5*(xcoord2[z_indx[i],i] - xcoord2[z_indx[i],i-1])
              right_x  = xcoord2[z_indx[i],i]
           vert_x.append(left_x)
           vert_x.append(right_x)

    # Appending the borders of the plot to close the patch
    vert_x.append(xcoord2[-1,-1])
    vert_x.append(xcoord2[-1,0])

    if vlevel == "MES":
       if max_dep == []:
          print("create_model_bathy_sec() ERROR:")
          print("With regular sigma levels you must provide")
          print("the maximum depth of the basin")
          return
       else:
          vert_z.append(max_dep)
          vert_z.append(max_dep)
    else:
       vert_z.append(tdep2[-1,-1] + 0.5*(tdep2[-1,-1]-tdep2[-2,-1]))
       vert_z.append(tdep2[-1,0] + 0.5*(tdep2[-1,0]-tdep2[-2,0]))

    return vert_x, vert_z
 
# =====================================================================================

def nemo_vgeom(vlevel, tdep3, wdep3):

    '''
    This function returns depth matrixes needed
    to plot correctly vertical levels and T points
    of NEMO model in sections plots. 
    '''

    ni = tdep3.shape[2]   # number of grid points along i-direction
    nj = tdep3.shape[1]   # number of grid points along j-direction
    nk = tdep3.shape[0]

    # Z PARTIAL STEPS
    if vlevel == "Z_ps":

       tpnt3 = np.copy(tdep3)
       wlev3 = np.zeros(shape=(nk,nj,ni))

       for k in np.arange(nk):

           wlev3[k,:,:] = np.unique(wdep3[k,:,:])[-1]

    # Z FULL STEPS or S / MES COORDINATES
    else:

       tpnt3 = np.copy(tdep3)
       wlev3 = np.copy(wdep3)

    return tpnt3, wlev3

# =====================================================================================

def nemo_ver_Tgrid4pcolor(var, xcoord, tgdep, wgdep):

    if var.ndim != 2:
       print("ERROR: check the dimensions of var variable!")
       return
    else:
       var2 = var

    if xcoord.ndim != 2:
       print("ERROR: check the dimensions of xcoord variable!")
       return

    if tgdep.ndim != 2:
       print("ERROR: check the dimensions of tgdep variable!")
       return

    if wgdep.ndim != 2:
       print("ERROR: check the dimensions of wgdep variable!")
       return

    # ==================
    #     xcoord 
    # ==================

    indx       = np.arange(1,xcoord.shape[1]+1) - 0.5
    indx_new   = np.arange(xcoord.shape[1]+1)
    xcoord_new = np.zeros(shape=(xcoord.shape[0]+1,len(indx_new)))

    for k in range(xcoord_new.shape[0]-1):
        f_x                = interp1d(indx, xcoord[k,:])
        xcoord_new[k,1:-1] = f_x(indx_new[1:-1])
        xcoord_new[k,0]    = xcoord[k,0] - (xcoord_new[k,1] - xcoord[k,0])
        xcoord_new[k,-1]   = xcoord[k,-1] + (xcoord[k,-1] - xcoord_new[k,-2])
    xcoord_new[-1,:]       = xcoord_new[-2,:]

    # ==================
    #      depth
    # ==================

    depth      = np.zeros(shape=(wgdep.shape[0]+1,len(indx_new)))

    wgdep1         = np.zeros(shape=(wgdep.shape[0]+1,len(indx)))
    wgdep1[:-1,:]  = wgdep
    wgdep1[-1,:]   = tgdep[-1,:] + (tgdep[-1,:] - wgdep[-1,:])

    xcoord1        = np.zeros(shape=(xcoord.shape[0]+1,len(indx)))
    xcoord1[:-1,:] = xcoord
    xcoord1[-1,:]  = xcoord[-1,:]

    # CREATING arrays for interpolation
    wgdep1_vec      = np.zeros(shape=(wgdep1.shape[0]*(wgdep1.shape[1]-1), 2 ))
    wgdep1_vec[:,0] = np.ravel(wgdep1[:,:-1], order='F')
    wgdep1_vec[:,1] = np.ravel(wgdep1[:,1:], order='F')

    xcoord1_vec      = np.zeros(shape=(wgdep1.shape[0]*(wgdep1.shape[1]-1), 2 ))
    xcoord1_vec[:,0] = np.ravel(xcoord1[:,:-1], order='F')
    xcoord1_vec[:,1] = np.ravel(xcoord1[:,1:], order='F')

    xcoord_4int     = xcoord_new[:,1:-1]
    xcoord_4int_vec = np.ravel(xcoord_4int, order='F')

    # LINEAR INTERPOLATION of wgdep on xcoord_new
    wgdep_int = wgdep1_vec[:,0] + \
                np.multiply( wgdep1_vec[:,1]-wgdep1_vec[:,0],\
                             np.divide(xcoord_4int_vec-xcoord1_vec[:,0],\
                                       xcoord1_vec[:,1]-xcoord1_vec[:,0]) 
                           )

    depth[:,1:-1] = np.reshape(wgdep_int, (wgdep1.shape[0],wgdep1.shape[1]-1), order='F')

    # EXTRAPOLATING depth values for left and right boundaries
    depth[:,0]    = depth[:,1]
    depth[:,-1]    = depth[:,-2]

    # ==================
    #     var
    # ==================

    mskval = 1.e-20

    var_new = np.ones(shape=(wgdep.shape[0]+1,len(indx_new))) * mskval
    var_new[:-1,:-1] = var
    var_new[var_new == mskval] = np.nan


    return var_new, xcoord_new, depth

