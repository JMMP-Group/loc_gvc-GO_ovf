#!/usr/bin/env python

#     |---------------------------------------------------------------|
#     | This module computes the initial density stratification of    |
#     | the ocean and of the cold water mass in the Denmark Strait    |
#     | for the idealised overflow experiment.                        |
#     | The method is similar to what was done in                     |
#     | Riemenschneider & Legg 2007, doi:10.1016/j.ocemod.2007.01.003 |
#     |                                                               |
#     | Author: Diego Bruciaferri                                     |
#     | Date and place: 03-12-2021, Met Office, UK                    |
#     |---------------------------------------------------------------|

import numpy as np
import gsw as gsw
import nsv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr

# ==============================================================================
# Input parameters

# 1. INPUT FILES
# Change this to match your local paths set-up
base_dir = "/your_local_path"
DOMCFG_dir = base_dir + '/models_geometry/dom_cfg/ideal'
TRACER_dir = base_dir + '/outputs/ideal_ovf/'


DOMCFG = [DOMCFG_dir + '/zps/domain_cfg_zps.nc',
          DOMCFG_dir + '/vqs/domain_cfg_vqs.nc',
          DOMCFG_dir + '/szt/domain_cfg_szt.nc',
          DOMCFG_dir + '/mes/domain_cfg_mes.nc',
         ]

exp_list = [[TRACER_dir + 'zps/nemo_cg602o_1d_19760101-19760201_grid-',
             TRACER_dir + 'szt/nemo_cg602o_1d_19760101-19760201_grid-',
             TRACER_dir + 'MEs/nemo_cg602o_1d_19760101-19760201_grid-',
            ],
            [TRACER_dir + 'zps/nemo_cg602o_1d_19760201-19760301_grid-',
              TRACER_dir + 'szt/nemo_cg602o_1d_19760201-19760301_grid-',
              TRACER_dir + 'MEs/nemo_cg602o_1d_19760201-19760301_grid-',
            ]]

exp_col = ['red','blue','green']
exp_lab = ['z*ps','szt','MEs']

# profile coordinates
I = [84, 67]
J = [77, 77]
T = [11, 0]

days = [12, 31]

for day in range(2):

    fig1  = plt.figure(figsize=(6,10))
    fig2  = plt.figure(figsize=(6,10))
    spec1 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig1)
    spec2 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig2)
    ax1   = fig1.add_subplot(spec1[:1])
    ax2   = fig2.add_subplot(spec2[:1])

    i = I[day]
    j = J[day]
    t = T[day]

    # ========================================================================
    for exp in range(len(exp_list[day])):

        # Loading domain geometry
        ds_dom  = xr.open_dataset(DOMCFG[exp])

        e3t = ds_dom["e3t_0"].squeeze().values[:,j,i]
        e3w = ds_dom["e3w_0"].squeeze().values[:,j,i]
        glamt = ds_dom["glamt"].squeeze().values[j,i]
        gphit = ds_dom["gphit"].squeeze().values[j,i]
        print("lon: ",glamt, "lat: ", gphit)
         
        nk = e3t.shape[0]

        # Computing model levels' depth
        tdep = np.zeros(shape=(nk,))
        wdep = np.zeros(shape=(nk,))
        wdep[0] = 0.
        tdep[0] = 0.5 * e3w[0]
        for k in range(1, nk):
            wdep[k] = wdep[k-1] + e3t[k-1]
            tdep[k] = tdep[k-1] + e3w[k]

        # Loading NEMO files
        ds_U = xr.open_dataset(exp_list[day][exp]+"U_cut.nc")
        ds_V = xr.open_dataset(exp_list[day][exp]+"V_cut.nc") 

        # Extracting the profile we want
        U =  ds_U['uo'].values[t,:,j,i]
        V =  ds_V['vo'].values[t,:,j,i]

        #=====================================================================================

        ax1.plot(U, tdep, exp_col[exp], 
                          linewidth=2, 
                          marker='o', 
                          markerfacecolor='black',
                          markersize=5,
                          label=exp_lab[exp])
        ax2.plot(V, tdep, exp_col[exp], 
                          linewidth=2,
                          marker='o', 
                          markerfacecolor='black',
                          markersize=5,
                          label=exp_lab[exp])

    ax1.invert_yaxis()
    ax1.grid(True)
    ax1.set_ylabel('Depth [$m$]', fontsize='25',color="black")
    ax1.set_xlabel('Across-slope velocity [$m\;s^{-1}$]', fontsize='25',color="black")
    ax1.tick_params(axis='y', labelsize=25)
    ax1.tick_params(axis='x', labelsize=25, which='major', width=1.50, length=10, labelcolor="black")

    ax2.invert_yaxis()
    ax2.grid(True)
    ax2.set_ylabel('Depth [$m$]', fontsize='25',color="black")
    ax2.set_xlabel('Along-slope velocity [$m\;s^{-1}$]', fontsize='25',color="black")
    ax2.tick_params(axis='y', labelsize=25)
    ax2.tick_params(axis='x', labelsize=25, which='major', width=1.50, length=10, labelcolor="black")

    #plt.rc('legend', **{'fontsize':30})
    fig_name = 'U_profile_'+str(days[day])+'.png'
    fig1.savefig(fig_name, bbox_inches="tight")
    fig_name = 'V_profile_'+str(days[day])+'.png'
    fig2.savefig(fig_name, bbox_inches="tight")
    print("done")

    plt.close()

