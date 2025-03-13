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
base_dir = "/data/users/dbruciaf/loc_gvc-nordic_ovf_DATA" #"/your_local_path"
DOMCFG_dir = base_dir + '/models_geometry/dom_cfg/ideal'
TRACER_dir = "/scratch/dbruciaf/ideal_ovf/" #base_dir + '/outputs/ideal_ovf/'


DOMCFG = [DOMCFG_dir + '/zps/domain_cfg_zps_cut.nc',
          DOMCFG_dir + '/szt/domain_cfg_szt_cut.nc',
          DOMCFG_dir + '/mes/domain_cfg_mes_cut.nc',
         ]

exp_list = [[TRACER_dir + 'zps/nemo_cg602o_1h_19760107-19760107_grid-',
             TRACER_dir + 'szt/nemo_cg602o_1h_19760107-19760107_grid-',
             TRACER_dir + 'MEs/nemo_cg602o_1h_19760107-19760107_grid-',
            ],
            [TRACER_dir + 'zps/nemo_cg602o_1h_19760114-19760114_grid-',
             TRACER_dir + 'szt/nemo_cg602o_1h_19760114-19760114_grid-',
             TRACER_dir + 'MEs/nemo_cg602o_1h_19760114-19760114_grid-',
            ]]

exp_col = ['red','blue','green']
exp_lab = ['z*ps','szt','MEs']

# profile coordinates
uI = [86,82]
uJ = [82,78]
vI = [87,82]
vJ = [81,77]
T = [23, 20]

days = [7, 14]

for day in range(2):

    fig1  = plt.figure(figsize=(6,12))
    fig2  = plt.figure(figsize=(6,12))
    spec1 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig1)
    spec2 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig2)
    ax1   = fig1.add_subplot(spec1[:1])
    ax2   = fig2.add_subplot(spec2[:1])

    iu = uI[day]
    ju = uJ[day]
    iv = vI[day]
    jv = vJ[day]
    t  = T[day]

    # ========================================================================
    for exp in range(len(exp_list[day])):

        # Loading domain geometry
        ds_dom  = xr.open_dataset(DOMCFG[exp])

        e3u = ds_dom["e3u_0"].squeeze().values[:,ju,iu]
        e3uw = ds_dom["e3uw_0"].squeeze().values[:,ju,iu]
        e3v = ds_dom["e3v_0"].squeeze().values[:,jv,iv]
        e3vw = ds_dom["e3vw_0"].squeeze().values[:,jv,iv]

        glamu = ds_dom["glamu"].squeeze().values[ju,iu]
        gphiu = ds_dom["gphiu"].squeeze().values[ju,iu]
        print("lon: ",glamu, "lat: ", gphiu)
         
        nk = e3u.shape[0]

        # Computing model levels' depth
        udep = np.zeros(shape=(nk,))
        uwdep = np.zeros(shape=(nk,))
        uwdep[0] = 0.
        udep[0] = 0.5 * e3uw[0]
        for k in range(1, nk):
            uwdep[k] = uwdep[k-1] + e3u[k-1]
            udep[k]  = udep[k-1] + e3uw[k]

        vdep = np.zeros(shape=(nk,))
        vwdep = np.zeros(shape=(nk,))
        vwdep[0] = 0.
        vdep[0] = 0.5 * e3vw[0]
        for k in range(1, nk):
            vwdep[k] = vwdep[k-1] + e3v[k-1]
            vdep[k]  = vdep[k-1] + e3vw[k]


        # Loading NEMO files
        ds_U = xr.open_dataset(exp_list[day][exp]+"U.nc")
        ds_V = xr.open_dataset(exp_list[day][exp]+"V.nc") 

        # Extracting the profile we want
        U =  ds_U['uo'].values[t,:,ju,iu]
        V =  ds_V['vo'].values[t,:,jv,iv]

        #=====================================================================================

        ax1.plot(U, udep, exp_col[exp], 
                          linewidth=2, 
                          marker='o', 
                          markerfacecolor='black',
                          markersize=5,
                          label=exp_lab[exp])
        ax2.plot(V, vdep, exp_col[exp], 
                          linewidth=2,
                          marker='o', 
                          markerfacecolor='black',
                          markersize=5,
                          label=exp_lab[exp])

    ax1.invert_yaxis()
    ax1.grid(True)
    ax1.set_ylabel('Depth [$m$]', fontsize='25',color="black")
    ax1.set_xlabel('Across-slope velocity [$m\;s^{-1}$]', fontsize='22',color="black")
    ax1.tick_params(axis='y', labelsize=25)
    ax1.tick_params(axis='x', labelsize=25, which='major', width=1.50, length=10, labelcolor="black")

    ax2.invert_yaxis()
    ax2.grid(True)
    ax2.set_ylabel('Depth [$m$]', fontsize='25',color="black")
    ax2.set_xlabel('Along-slope velocity [$m\;s^{-1}$]', fontsize='22',color="black")
    ax2.tick_params(axis='y', labelsize=25)
    ax2.tick_params(axis='x', labelsize=25, which='major', width=1.50, length=10, labelcolor="black")

    #plt.rc('legend', **{'fontsize':30})
    fig_name = 'U_profile_hr_'+str(days[day])+'.png'
    fig1.savefig(fig_name, bbox_inches="tight")
    fig_name = 'V_profile_hr_'+str(days[day])+'.png'
    fig2.savefig(fig_name, bbox_inches="tight")
    print("done")

    plt.close()

