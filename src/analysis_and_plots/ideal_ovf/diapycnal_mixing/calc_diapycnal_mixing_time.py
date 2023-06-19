#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import os
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from xnemogcm import open_domain_cfg, open_nemo
from geopy.distance import great_circle
import matplotlib.colors as colors
from scipy.stats import binned_statistic_2d

# ==============================================================================
# Input parameters

# 1. INPUT FILES

# Change this to match your local paths set-up
base_dir = "/your_local_path"

exp_list = ["zps", "szt", "mes"]

domcfg_base = base_dir + '/models_geometry/dom_cfg/ideal'

DOMCFG_list = [domcfg_base + '/zps/domain_cfg_zps.nc',
               domcfg_base + '/szt/domain_cfg_szt.nc',
               domcfg_base + '/mes/domain_cfg_mes.nc']

comm_dir = base_dir + '/ideal_ovf'

file1 = "nemo_cg602o_1d_19760101-19760201_grid_T.nc"
file2 = "nemo_cg602o_1d_19760201-19760301_grid_T.nc"
file3 = "nemo_cg602o_1d_19760301-19760401_grid_T.nc"
file4 = "nemo_cg602o_1d_19760401-19760501_grid_T.nc"

T_list = [[comm_dir+'/zps/'+file1,
           comm_dir+'/zps/'+file2,
           comm_dir+'/zps/'+file3,
           comm_dir+'/zps/'+file4],
          [comm_dir+'/szt/'+file1,
           comm_dir+'/szt/'+file2,
           comm_dir+'/szt/'+file3,
           comm_dir+'/szt/'+file4],
          [comm_dir+'/MEs/'+file1,
           comm_dir+'/MEs/'+file2,
           comm_dir+'/MEs/'+file3,
           comm_dir+'/MEs/'+file4]]

# 2. ANALYSIS
tra_lim = 0.1 # minimum passive tracer [] -> to identify dense plume 
ovf_ini_i = 1043 
ovf_ini_j = 1017

# ==============================================================================

for exp in range(len(exp_list)):

    print(exp_list[exp])

    # Loading domain geometry
    ds_dom  = open_domain_cfg(files=[DOMCFG_list[exp]])
    for i in ['bathymetry','bathy_meter']:
        for dim in ['x','y']:
            ds_dom[i] = ds_dom[i].rename({dim: dim+"_c"})

    # Loading NEMO files
    ds_tra = open_nemo(ds_dom, files=T_list[exp])

    # Extracting only the part of the domain we need
    x1 = 985
    x2 = 1055
    y1 = 980
    y2 = 1020
    dsD = ds_dom.isel(x_c=slice(x1,x2),x_f=slice(x1,x2),y_c=slice(y1,y2),y_f=slice(y1,y2))
    dsT = ds_tra.isel(x_c=slice(x1,x2),y_c=slice(y1,y2))

    # Creating time array 
    ni = len(dsD.x_c)
    nj = len(dsD.y_c)
    nk = len(dsD.z_c)
    nt = len(dsT.t)
    print(nt,nj,ni)
    time = np.zeros(shape=(nt,nk,nj,ni))
    for t in range(nt):
        time[t,:,:,:] = t+1

    e1t = dsD["e1t"].expand_dims({"z_c": dsD.z_c})
    e2t = dsD["e2t"].expand_dims({"z_c": dsD.z_c})
    e1t = e1t.expand_dims({"t": dsT.t})
    e2t = e2t.expand_dims({"t": dsT.t})

    # Extracting DataArrays
    e3t  = dsT['thkcello'].fillna(0.)
    C = dsT['so_seos'] - 20. # Concentration as mass fraction g/kg
    C = C.where(C>0.,0.).fillna(0.)
    T = dsT['thetao_seos'].fillna(0.)

    # Computing density
    rho0 = 1026. # kg/m3
    a0   = 5.e-1 #1.6550e-1
    RHO = rho0 - a0*(T-10.)      

    CV = e1t * e2t * e3t # cells' volume
    WM = RHO * CV        # Water mass in kg
    PT = WM * C          # Amount of passive tracer in g in each cell 

    x_bins = np.linspace(0, nt, 31)   # time - every 4d
    y_bins = np.linspace(27.6, 28.8, 21)  # density

    T1d = time.ravel(order='C')
    P1d = PT.values.ravel(order='C')
    R1d = RHO.values.ravel(order='C')
    H, xedges, yedges, binnumber = binned_statistic_2d(T1d, 
                                                       R1d, 
                                                       P1d, 
                                                       statistic="sum", 
                                                       bins=[x_bins, y_bins])

    #print("H", H, H.shape)
    #print("xedges", xedges, xedges.shape)
    #print("yedges", yedges, yedges.shape)

    # create dataset
    data_vars = {'H':(['time','density'], 
                      H,
                      {'units': 'kg',
                       'long_name': 'Tot. amount of passive tracer per density class in 4 days'
                      }   
                     )
                }
    coords = {'time': (['time'], 
                       xedges[1::],
                       {'start': '0'}
                      ),
              'density': (['density'], 
                          yedges[1::],
                          {'start': '27.6'}
                         ),
             }
    attrs = {'author':'Diego Bruciaferri', 
             'email':'diego.bruciaferri@metoffice.gov.uk'}

    ds = xr.Dataset(data_vars=data_vars, 
                    coords=coords, 
                    attrs=attrs) 

    ds.to_netcdf(exp_list[exp] + '_dia_mixing_time_4d.nc')
    print("done")

