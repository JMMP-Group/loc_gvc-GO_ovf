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
for obs in range(5):

    if   obs == 0:
       ds_obs = nsv.Standardizer().latrabjarg_climatology
       ds_obs = ds_obs.expand_dims({"time":[0]})
       label = "latrabjarg"
    elif obs == 1:
       ds_obs = nsv.Standardizer().pos503(5).drop_vars('time')
       ds_obs = ds_obs.expand_dims({"time":[0]})
       label = "pos503-5"
    elif obs == 2:
       ds_obs = nsv.Standardizer().ho2000
       ds_obs = ds_obs.expand_dims({"time":[0]})
       label = "ho2000"
    elif obs == 3:
       ds_obs = nsv.Standardizer().osnap
       ds_obs = ds_obs.isel(station=slice(80, None)) # only osnap east
       ds_obs = ds_obs.where(ds_obs.longitude<=-31.2) # only Irminger Sea
       label = "osnap-IS"
       ds_obs = xr.decode_cf(ds_obs)
    elif obs == 4:
       ds_obs = nsv.Standardizer().osnap
       ds_obs = ds_obs.isel(station=slice(80, None)) # only osnap east
       ds_obs = ds_obs.where(np.logical_and(ds_obs.longitude>-31.2,
                                            ds_obs.longitude<=-13.2)
                            )  # only Icelandic Basin
       label = "osnap-IB"
       ds_obs = xr.decode_cf(ds_obs)

    if obs < 3: 
       # DSOW @ DS: 
       # 1) Dickson and Brown 1994
       # 2) Osterhus et al. 2019 (https://doi.org/10.5194/os-15-379-2019)
       rho_ovf = 27.80
    elif obs == 3:
       # DSOW @ Irminger sea 
       #T_ovf = 3.0
       rho_ovf = 27.84
    elif obs == 4:
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

    # Computing TEOS-10 T,S and rho
    ds_obs = compute_teos10(ds_obs)
    rho = ds_obs['sigma_theta'] # potential density referred at 0m
    tem = ds_obs["potential_temperature"]
    sal = ds_obs["practical_salinity"]
    
    # Interpolating in the middle of the cell
    rho_data = 0.5 * (rho.isel(z=slice(1,None) , x=slice(1,None) ).values + \
                      rho.isel(z=slice(None,-1), x=slice(None,-1)).values)
    tem_data = 0.5 * (tem.isel(z=slice(1,None) , x=slice(1,None) ).values + \
                      tem.isel(z=slice(None,-1), x=slice(None,-1)).values)
    sal_data = 0.5 * (sal.isel(z=slice(1,None) , x=slice(1,None) ).values + \
                      sal.isel(z=slice(None,-1), x=slice(None,-1)).values)
    rho = xr.DataArray(rho_data,dims=rho.dims)
    tem = xr.DataArray(tem_data,dims=tem.dims)
    sal = xr.DataArray(sal_data,dims=sal.dims)
    area = xr.DataArray(area,dims=sal.dims)
  
    rho  = rho.where(rho>rho_ovf)
    tem  = tem.where(rho>rho_ovf)
    sal  = sal.where(rho>rho_ovf)
    area = area.where(rho>rho_ovf)

    print(label + ': computing mean T, S and RHO of the ovf')
    rho_avg = (rho*area).sum(dim=('z','x'),skipna=True) / area.sum(dim=('z','x'),skipna=True)
    tem_avg = (tem*area).sum(dim=('z','x'),skipna=True) / area.sum(dim=('z','x'),skipna=True)
    sal_avg = (sal*area).sum(dim=('z','x'),skipna=True) / area.sum(dim=('z','x'),skipna=True)

    
    ds_ovf = xr.Dataset(
                  data_vars=dict(
                     rho_avg=(["t"], rho_avg.data),
                     tem_avg=(["t"], tem_avg.data),
                     sal_avg=(["t"], sal_avg.data),
                  ),
                  coords=dict(
                     time=(["t"], ds_obs.time.data)
                  ),
                  attrs=dict(description="average temperature, salinity and potenital density of overflows"),
             )

    ds_ovf.to_netcdf(label+'_ovf_properties.nc')

