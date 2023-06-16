#!/usr/bin/env python

import numpy as np
from numba import jit, float64
from scipy import interpolate
import xarray as xr
import nsv
import gsw

def compute_potential_sigma(ds):
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

    return ds

def depth2rho(vflux, rho):
    bins      = np.arange(23.295,28.115,0.01)
    bins_sect = (bins[1:]+bins[:-1])/2

    vflux_rho = rho_bin_loop(vflux, rho, bins)   
 
    vflux_rho = xr.DataArray(
                     data=vflux_rho,
                     dims=('t','x','rho_bin'),
                     name='vflux_rho'
                )  

    return bins_sect, vflux_rho      

#@jit(nopython=True)
def rho_bin_loop(vflux, rho, bins):
    nt  = vflux.shape[0]
    nx  = vflux.shape[2]
    nbins = len(bins)

    vflux_rho = np.zeros((nt,nx,nbins-1))

    #print(vflux.dtype, rho.dtype, bins.dtype, vflux_rho.dtype)

    for t in range(nt):
        print('time-step', t)
        for i in range(nx):
            for r in range(nbins-1):
                indexes = np.where((rho[t,:,i]>=bins[r])&(rho[t,:,i]<bins[r+1]))
                #tmp     = np.nansum(vflux[t,indexes,i])
                vflux_rho[t,i,r] = np.nansum(vflux[t,indexes,i])

    return vflux_rho

#=======================================================================================
# Input parameters

# Change this to match your local paths set-up
base_dir = "/your_local_path"


exp = [base_dir + "/loc_gvc-nordic_ovf/outputs/realistic/zps/nemo_cn092o_20140101-20181001_osnap.nc",
       base_dir + "/loc_gvc-nordic_ovf/outputs/realistic/szt/nemo_cn093o_20140101-20181201_osnap.nc",
       base_dir + "/loc_gvc-nordic_ovf/outputs/realistic/MEs/nemo_cn094o_20140101-20181201_osnap.nc"
      ]
lab = ["zps", "szt", "MEs"]

# ------------------------------------------------------------------------------
# COMPUTING MOC IN DENSITY SPACE FROM OSNAP OBSERVATIONS

ds_obs = nsv.Standardizer().osnap
ds_obs = ds_obs.isel(station=slice(80, None)) # only osnap east
ds_obs = ds_obs.where(ds_obs.longitude<-31.2)  # only DS
ds_obs = ds_obs.rename_dims({'time':'t','depth':'z','station':'x'})
ds_obs = ds_obs.transpose("t", "z", "x")

# GRID
depth_o = ds_obs.depth.values
dz      = depth_o[1:]-depth_o[:-1]
dist    = ds_obs.distance.values*1000.
dx      = dist[1:]-dist[:-1]
# Compute the area
xx, zz  = np.meshgrid(dx,dz)
area    = zz * xx

# Normal Velocity
vnorm  = ds_obs.velo
v_sect = 0.5 * (vnorm.isel(z=slice(1,None),x=slice(1,None)).values + \
                vnorm.isel(z=slice(None,-1),x=slice(None,-1)).values)
v_sect = xr.DataArray(v_sect,dims=vnorm.dims)

# Mask the area
mask   = np.ma.masked_invalid(v_sect.isel(t=[0])).mask.squeeze()
area   = np.ma.masked_where(mask, area)
area   = np.repeat(area[np.newaxis,:,:], v_sect.t.size, axis=0)

# Vflux
Vflux = v_sect * area

# Potential density
ds_obs = compute_potential_sigma(ds_obs)
sigma0 = ds_obs.sigma_theta
s_sect = 0.5 * (sigma0.isel(z=slice(1,None),x=slice(1,None)).values + \
                sigma0.isel(z=slice(None,-1),x=slice(None,-1)).values)
s_sect = xr.DataArray(s_sect,dims=sigma0.dims)

# Transforming from depth to sigma space
bins_sect, vflux_rho = depth2rho(vflux=Vflux.values, rho=s_sect.values)

# Compute MOC from flux in density bins
print('Compute the MOC in sigma0 coordinates')
VFLUX_rho = np.zeros((vflux_rho.t.size, bins_sect.size))
for t in range(vflux_rho.t.size):
    # Integrate along x
    VFLUX_rho[t,:] = vflux_rho[t,:,:].sum(dim='x').values
    
ds_vflux = xr.Dataset(
              data_vars=dict(
                    vflux_rho=(["t", "rho_bins"], VFLUX_rho.data)
              ),
              coords=dict(
                    time=(["t"], ds_obs.time.data),
                    rho_bins=(["rho_bins"], bins_sect),
              ),
              attrs=dict(description="transport sigma space in the OSNAP_EAST Irminger basin section"),
         )

ds_vflux.to_netcdf('vflux_rho-DS_obs.nc')

# ----------------------------------------------------------------------------
# COMPUTING MOC IN SIGMA COORDS ALONG OSNAP SECTIONS FROM NEMO MODEL OUTPUTS

for e in range(len(exp)):

    ds = xr.open_dataset(exp[e]).squeeze()
    ds = ds.rename_dims({'time_counter':'t','deptht':'z'})
    time_m = ds.time_centered
    ds = ds.where(ds.nav_lon<-31.2) # only DS
    #ds = ds_obs.transpose("t", "z", "x")

    vnorm   = ds.vo.values
    t_con   = ds.thetao_con.values 
    s_abs   = ds.so_abs.values
    depth_m = ds.depu3d.values
    xx_m    = ds.e1v.values         # nt x nx
    zz_m    = ds.e3v_0.values       # nt x nz x nx

    nt = vnorm.shape[0]
    nk = vnorm.shape[1]
    ni = vnorm.shape[2]
    
    xx_m   = np.repeat(xx_m[:,np.newaxis,:], nk, axis=1)
    area_m = zz_m * xx_m

    # Mask the area
    mask_m = np.ma.masked_invalid(vnorm).mask
    area_m = np.ma.masked_where(mask_m, area_m)
 
    # Vflux
    Vflux = vnorm * area_m

    # Potential density
    s_sect = gsw.density.sigma0(s_abs, t_con)
  
    # Transforming from depth to sigma space
    bins_sect, vflux_rho = depth2rho(vflux=Vflux, rho=s_sect)

    # Compute flux in density bins
    print('Compute flux in sigma0 coordinates')
    VFLUX_rho = np.zeros((vflux_rho.t.size, bins_sect.size))
    for t in range(vflux_rho.t.size):
        # Integrate along x
        VFLUX_rho[t,:] = vflux_rho[t,:,:].sum(dim='x').values
       
    ds_vflux = xr.Dataset(
                  data_vars=dict(
                        vflux_rho=(["t", "rho_bins"], VFLUX_rho.data)
                  ),
                  coords=dict(
                        time=(["t"], time_m.data),
                        rho_bins=(["rho_bins"], bins_sect),
                  ),
                  attrs=dict(description="transport sigma space in the OSNAP_EAST Irminger basin section"),
             )

    ds_vflux.to_netcdf('vflux_rho-DS_' + lab[e] + '.nc')
