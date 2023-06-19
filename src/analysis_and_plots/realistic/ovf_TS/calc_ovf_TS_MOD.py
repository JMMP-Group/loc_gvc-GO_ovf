#!/usr/bin/env python

import numpy as np
import glob
from scipy import interpolate
import xarray as xr
import nsv
import gsw

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

# ================================================================================

# 1. INPUT FILES
# Change this to match your local paths set-up
base_dir = "/your_local_path"

domcfg = [base_dir + "/models_geometry/dom_cfg/realistic/zps/domain_cfg_zps.nc",
          base_dir + "/models_geometry/dom_cfg/realistic/szt/domain_cfg_szt.nc",
          base_dir + "/models_geometry/dom_cfg/realistic/mes/domain_cfg_mes.nc"]
inpdir =  "/local/path_to/monthly/model_data"
exp = ['zps','szt','MEs']

for obs in range(3,5):

    if   obs == 0:
       ds_obs = nsv.Standardizer().latrabjarg_climatology
       label = "latrabjarg"
    elif obs == 1:
       ds_obs = nsv.Standardizer().pos503(5).drop_vars('time')
       label = "pos503-5"
    elif obs == 2:
       ds_obs = nsv.Standardizer().ho2000
       label = "ho2000"
    elif obs == 3:
       ds_obs = nsv.Standardizer().osnap
       ds_obs = ds_obs.isel(station=slice(80, None)) # only osnap east
       label = "osnap-IS"
       ds_obs = xr.decode_cf(ds_obs)
    elif obs == 4:
       ds_obs = nsv.Standardizer().osnap
       ds_obs = ds_obs.isel(station=slice(80, None)) # only osnap east
       label = "osnap-IB"
       ds_obs = xr.decode_cf(ds_obs)

    print(label)

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

    # Extracting section
    obs_lon = ds_obs.longitude
    obs_lat = ds_obs.latitude

    # Managing OSNAP section
    if obs == 3:
       obs_lon = obs_lon.where(obs_lon<=-31.2) # Irminger Sea
       obs_lat = obs_lat.where(obs_lon<=-31.2)
    elif obs == 4:
       obs_lon = obs_lon.where(np.logical_and(obs_lon>-31.2, 
                                              obs_lon<=-13.2)
                              ) # Icelandic Basin
       obs_lat = obs_lat.where(np.logical_and(obs_lon>-31.2, 
                                              obs_lon<=-13.2)
                              )

    obs_lon = obs_lon.dropna("station")
    obs_lat = obs_lat.dropna("station")

    del ds_obs

    for n in range(len(exp)):

        inp_dir = inpdir + '/' + exp[n]
        
        ds_dom = xr.open_dataset(domcfg[n]).squeeze()
        # Computing model levels depth if needed
        if not ("gdept_0" in ds_dom and "gdepw_0" in ds_dom):
           ds_dom = compute_levels(ds_dom, merge=True)

        # Extracting section
        finder = nsv.SectionFinder(ds_dom)

        stations = finder.nearest_neighbor(
                      lons = obs_lon, 
                      lats = obs_lat,
                      grid="t"
                   )

        ds_dom  = ds_dom.isel({dim: stations[f"{dim}_index"] for dim in ("x", "y")})
        gphit   = ds_dom.gphit
        glamt   = ds_dom.glamt
        e1t     = ds_dom.e1t
        e2t     = ds_dom.e2t
        e1t     = e1t.expand_dims({"z": ds_dom.z})
        e2t     = e2t.expand_dims({"z": ds_dom.z})
        gphit   = gphit.expand_dims({"z": ds_dom.z})
        glamt   = glamt.expand_dims({"z": ds_dom.z})
        gdept_0 = ds_dom.gdept_0

        del ds_dom          

        Tlist = sorted(glob.glob(inp_dir + '/*grid-T.nc'))

        rho_avg = []
        tem_avg = []
        sal_avg = []
        time    = []

        for Tfile in Tlist:
            ds = xr.open_dataset(Tfile)
            ds = ds.rename_dims({'deptht':'z'})
            ds = ds.reset_index(["deptht"]).reset_coords(["deptht"])
            ds = ds.isel({dim: stations[f"{dim}_index"] for dim in ("x", "y")})
            
            e3t  = ds.thkcello
            # Computing potential temperature
            tem = gsw.conversions.pt_from_CT(ds.so_abs, ds.thetao_con)
            # Computing practical salinity
            prs = gsw.p_from_z(-gdept_0, gphit)
            sal = gsw.SP_from_SA(ds.so_abs, prs, glamt, gphit)
            # Computing potential density anomaly
            rho = gsw.density.sigma0(ds.so_abs, ds.thetao_con)

            vol = e1t * e2t * e3t 

            rho = rho.where(rho>rho_ovf)
            tem = tem.where(rho>rho_ovf)
            sal = sal.where(rho>rho_ovf)
            vol = vol.where(rho>rho_ovf)

            print(label + ': computing mean T, S and RHO of the ovf')
            rho_avg.append(((rho*vol).sum(dim=('z','station'),skipna=True) / vol.sum(dim=('z','station'),skipna=True)).values[0])
            tem_avg.append(((tem*vol).sum(dim=('z','station'),skipna=True) / vol.sum(dim=('z','station'),skipna=True)).values[0])
            sal_avg.append(((sal*vol).sum(dim=('z','station'),skipna=True) / vol.sum(dim=('z','station'),skipna=True)).values[0])
            #time.append(ds.indexes['time_counter'].to_datetimeindex())
            time.append(ds.time_counter.values[0])   

        ds_ovf = xr.Dataset(
                       data_vars=dict(
                            rho_avg=(["t"], rho_avg),
                            tem_avg=(["t"], tem_avg),
                            sal_avg=(["t"], sal_avg),
                       ),
                       coords=dict(
                            time=(["t"], time)
                       ),
                       attrs=dict(description="average temperature, salinity and potenital density of overflows"),
                 )

        ds_ovf.to_netcdf(exp[n] + "_" + label + '_ovf_properties.nc')

