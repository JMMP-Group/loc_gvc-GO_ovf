#!/usr/bin/env python

import numpy as np
import glob
import xarray as xr
import pandas as pd
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
# Input parameters

# Change this to match your local paths set-up
base_dir = "/your_local_path"
base_inp = "/your/local/path/to/monthly/model/data"

domcfg = [base_dir + "/models_geometry/dom_cfg/realistic/zps/domain_cfg_zps.nc",
          base_dir + "/models_geometry/dom_cfg/realistic/szt/domain_cfg_szt.nc",
          base_dir + "/models_geometry/dom_cfg/realistic/mes/domain_cfg_mes.nc"]
exp = ['zps','szt','MEs']

for obs in range(6,7):

    if obs == 0:
       ds_obs = nsv.Standardizer().osnap
       ds_obs = ds_obs.isel(station=slice(80, None)) # only osnap east
       label = "osnap-IS"
       ds_obs = xr.decode_cf(ds_obs)
    elif obs == 1:
       ds_obs = nsv.Standardizer().osnap
       ds_obs = ds_obs.isel(station=slice(80, None)) # only osnap east
       label = "osnap-IB"
       ds_obs = xr.decode_cf(ds_obs)
    elif obs == 2:
       ds_obs = nsv.Standardizer().latrabjarg_climatology 
       label = "latrabjarg"
       ds_obs = xr.decode_cf(ds_obs)
    elif obs == 3:
       ds_obs = nsv.Standardizer().osnap
       lon = [-12.5,-9.8 ]
       lat = [64.38, 63.1]
       label = "ifr"
    elif obs == 4:
       ds_obs = nsv.Standardizer().osnap
       lon = [-7.49, -8.36] 
       lat = [61.64, 61.18]
       label = "fbc"
    elif obs == 5:
       ds_obs = nsv.Standardizer().osnap
       lon = [-9.5, -8.2 ]
       lat = [60.65, 59.9 ]
       label = "wtr"
    elif obs == 6:
       ds_obs = nsv.Standardizer().osnap
       lon = [-35.33, -35.33]
       lat = [52.925, 52.118]
       label = "cgfz"

    print(label)

    if obs == 0:
       # DSOW @ Irminger sea 
       #T_ovf = 3.0
       rho_ovf = 27.84
    elif obs == 1:
       # ISOW @ Icelandic sea 
       rho_ovf = 27.84
    elif obs == 6:
       rho_ovf = 27.84
    else:
       rho_ovf = 27.80

    # Extracting section  
    obs_lon = ds_obs.longitude
    obs_lat = ds_obs.latitude

    # Managing OSNAP section
    if obs == 0:
       obs_lon = obs_lon.where(obs_lon<=-31.2) # Irminger Sea
       obs_lat = obs_lat.where(obs_lon<=-31.2)
    elif obs == 1:
       obs_lon = obs_lon.where(np.logical_and(obs_lon>-31.2, 
                                              obs_lon<=-13.2)
                              ) # Icelandic Basin
       obs_lat = obs_lat.where(np.logical_and(obs_lon>-31.2, 
                                              obs_lon<=-13.2)
                              )

    # Managing fbc and wtr sections
    if obs > 2 :
       obs_lon[:] = np.nan
       obs_lat[:] = np.nan
       obs_lon[0:len(lon)] = lon
       obs_lat[0:len(lat)] = lat

    obs_lon = obs_lon.dropna("station")
    obs_lat = obs_lat.dropna("station")

    del ds_obs

    for n in range(len(exp)):

        inp_dir = base_inp + '/' + exp[n]
        model = exp[n][0:3]
        print(model)

        ds_dom = xr.open_dataset(domcfg[n]).squeeze()
        # Computing model levels depth if needed
        if not ("gdept_0" in ds_dom and "gdepw_0" in ds_dom):
           ds_dom = compute_levels(ds_dom, merge=True)

        # Extracting section
        finder = nsv.SectionFinder(ds_dom)

        stations_UV = finder.velocity_points_along_zigzag_section(
                             lons=obs_lon,
                             lats=obs_lat
                      )
        stations_T  = finder.zigzag_section(lons=obs_lon,
                             lats=obs_lat,
                             grid='t'
                      )

        #print(stations_UV)
        #print(stations_T)

        if 'u' in stations_UV:
           xU = stations_UV['u'].x_index
           yU = stations_UV['u'].y_index
        else:
           xU = yU = None
        if 'v' in stations_UV:
           xV = stations_UV['v'].x_index
           yV = stations_UV['v'].y_index
        else:
           xV = yV = None
        xT = stations_T.x_index
        yT = stations_T.y_index

        if xU is not None:
           glamu = ds_dom["glamu"].isel(x=xU, y=yU)
           gphiu = ds_dom["gphiu"].isel(x=xU, y=yU)
           e1u   = ds_dom["e1u"].isel(x=xU, y=yU).expand_dims({"z": ds_dom.z}).rename("e1")
   
        if xV is not None:
           glamv = ds_dom["glamv"].isel(x=xV, y=yV)
           gphiv = ds_dom["gphiv"].isel(x=xV, y=yV)
           e2v   = ds_dom["e2v"].isel(x=xV, y=yV).expand_dims({"z": ds_dom.z}).rename("e1")

        if xU is None:
           e1u = e2v.copy()
        if xV is None:
           e2v = e1u.copy()
        
        ds_e1 = xr.merge([e1u, e2v], combine_attrs='drop_conflicts')
        e1 = ds_e1.e1

        del ds_dom          

        Tlist = sorted(glob.glob(inp_dir + '/*_1m_*grid-T.nc'))
        Ulist = sorted(glob.glob(inp_dir + '/*_1m_*grid-U.nc'))
        Vlist = sorted(glob.glob(inp_dir + '/*_1m_*grid-V.nc'))

        flx_tot = []
        time    = []

        for F in range(len(Tlist)):

            Tfile = Tlist[F]
            ds = xr.open_dataset(Tfile).squeeze()
            ds = ds.rename_dims({'deptht':'z'})
            ds = ds.reset_index(["deptht"]).reset_coords(["deptht"])
            ds = ds.isel({dim: stations_T[f"{dim}_index"] for dim in ("x", "y")})      
            e3t  = ds.thkcello
            # Computing potential density anomaly
            rho = gsw.density.sigma0(ds.so_abs, ds.thetao_con)
            del ds

            if xU is not None:
               Ufile = Ulist[F]
               ds = xr.open_dataset(Ufile).squeeze()
               ds = ds.rename_dims({'depthu':'z'})
               ds = ds.reset_index(["depthu"]).reset_coords(["depthu"])
               ds = ds.isel(x=xU, y=yU)
               uo = ds.uo.rename("vnorm")
               time_count = ds.time_counter
               del ds

            if xV is not None:
               Vfile = Vlist[F]
               ds = xr.open_dataset(Vfile).squeeze()
               ds = ds.rename_dims({'depthv':'z'})
               ds = ds.reset_index(["depthv"]).reset_coords(["depthv"])
               ds = ds.isel(x=xV, y=yV)
               vo = ds.vo.rename("vnorm")
               del ds

            if xU is None: uo = vo.copy()
            if xV is None: vo = uo.copy()

            rho = rho.rolling({'station':2}, center=True).mean()
            e3  = e3t.rolling({'station':2}, center=True).mean() 
            first = 0
            #print(e3.shape)
            #print(e1.shape)
            if e3.shape[1] != e1.shape[1]: first = np.absolute(e3.shape[1]-e1.shape[1])
            if e3.shape[1] > e1.shape[1]:
               rho = rho.isel(station=slice(first,None))
               e3  = e3.isel(station=slice(first,None))
            elif e3.shape[1] < e1.shape[1]:
               e1  = e1.isel(station=slice(first,None))

            area = e1 * e3

            ds_u = xr.merge([vo, uo], combine_attrs='drop_conflicts')
            vnorm = ds_u.vnorm

            Vflux = vnorm * area

            # Only OVF
            Vflux = Vflux.where(rho>rho_ovf)

            print(label + ': computing tot transport of the ovf, ' + str(time_count.values))
            flx_tot.append(Vflux.sum(dim=('z','station'),skipna=True))
            time.append(pd.to_datetime(str(time_count.values)))

            ds_ovf = xr.Dataset(
                           data_vars=dict(
                                Vflx_tot=(["t"], flx_tot),
                           ),
                           coords=dict(
                                time=(["t"], time)
                           ),
                           attrs=dict(description="tot volume flux for overflow waters with rho > " + str(rho_ovf),
                           ),
                     )

            ds_ovf.to_netcdf(model + "_" + label + '_ovf_vflux.nc')

