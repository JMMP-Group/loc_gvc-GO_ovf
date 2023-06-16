#!/usr/bin/env python

import numpy as np
from scipy import interpolate
import xarray as xr
import nsv
import gsw

#=======================================================================================
def compute_teos10(ds):
    # Absolute Salinity
    press = gsw.p_from_z(-ds.depth, ds.latitude)
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
    ds['cnsrv_temp']  = con_t
    ds['abslt_sali']  = abs_s

    return ds

#=======================================================================================
for obs in range(2):

    if obs == 0:
       ds_obs = nsv.Standardizer().osnap
       ds_obs = ds_obs.isel(station=slice(80, None)) # only osnap east
       ds_obs = ds_obs.where(ds_obs.longitude<=-31.2) # only Irminger Sea
       label = "osnap-IS"
       ds_obs = xr.decode_cf(ds_obs)
    elif obs == 1:
       ds_obs = nsv.Standardizer().osnap
       ds_obs = ds_obs.isel(station=slice(80, None)) # only osnap east
       ds_obs = ds_obs.where(np.logical_and(ds_obs.longitude>-31.2,
                                            ds_obs.longitude<=-13.2)
                            )  # only Icelandic Basin
       label = "osnap-IB"
       ds_obs = xr.decode_cf(ds_obs)

    if obs == 0:
       # DSOW @ Irminger sea 
       #T_ovf = 3.0
       rho_ovf = 27.84
    elif obs == 1:
       # ISOW @ Icelandic sea 
       rho_ovf = 27.84
    
    ds_obs = ds_obs.rename_dims({'time':'t','depth':'z','station':'x'})
    ds_obs = ds_obs.transpose("t", "z", "x")

    # GRID
    depth     = ds_obs.depth.values
    depth_out = ds_obs.depth.rolling(z=2).mean().dropna("z")
    dz        = depth[1:]-depth[:-1]
    dist      = ds_obs.distance.values*1000.
    dx        = dist[1:]-dist[:-1]
    # Compute the area
    xx, zz  = np.meshgrid(dx,dz)
    area    = zz * xx
    area    = np.repeat(area[np.newaxis,:,:], ds_obs.t.size, axis=0)

    # -----------------------------------------------------------------------------
    # Computing potential density
    ds_obs = compute_teos10(ds_obs)
    rho = ds_obs['sigma_theta'] # potential density referred at 0m
    
    # Interpolating in the middle of the cell
    rho_data = 0.5 * (rho.isel(z=slice(1,None) , x=slice(1,None) ).values + \
                      rho.isel(z=slice(None,-1), x=slice(None,-1)).values)
    rho = xr.DataArray(rho_data,dims=rho.dims)
    area = xr.DataArray(area,dims=rho.dims)
  
    # -----------------------------------------------------------------------------
    # Normal Velocity
    vnorm  = ds_obs.velo
    v_sect = 0.5 * (vnorm.isel(z=slice(1,None),x=slice(1,None)).values + \
                    vnorm.isel(z=slice(None,-1),x=slice(None,-1)).values)
    v_sect = xr.DataArray(v_sect,dims=vnorm.dims)

    # -----------------------------------------------------------------------------
    # Vflux
    Vflux = v_sect * area

    # -----------------------------------------------------------------------------
    # Only OVF
    Vflux = Vflux.where(rho>rho_ovf)
    
    print(label + ': computing tot transport of the ovf')
    Vflx_tot = Vflux.sum(dim=('z','x'),skipna=True) 
    
    ds_ovf = xr.Dataset(
                  data_vars=dict(
                     Vflx_tot=(["t"], Vflx_tot.data),
                  ),
                  coords=dict(
                     time=(["t"], ds_obs.time.data)
                  ),
                  attrs=dict(description="tot volume flux for overflow waters with rho > " + str(rho_ovf),
                  ),
             )

    ds_ovf.to_netcdf(label+'_ovf_vflux.nc')
