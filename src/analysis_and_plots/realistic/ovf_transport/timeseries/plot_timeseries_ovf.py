#!/usr/bin/env python

import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as mtick
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import AnchoredText
from matplotlib.gridspec import GridSpec

#=======================================================================================
# Input parameters

# Change this to match your local paths set-up
base_dir = "/your_local_path"


mod_dir = base_dir + "/realistic/"
obs_dir = base_dir + "/realistic/obs/ovf_transport/"
sections = ["latrabjarg","ifr","fbc","wtr","cgfz","osnap-IB","osnap-IS"]
cols = ["black","red","dodgerblue","limegreen"]

for sec in sections:

    print("====", sec)

    #fig = plt.figure(figsize = (32.,17.5), dpi=100, constrained_layout=True)
    fig = plt.figure(figsize = (22.,17.5), dpi=100, constrained_layout=True)

    if "osnap" in sec:

       obs = True
       # OBS
       File = obs_dir + sec + "_ovf_vflux.nc"
       ds   = xr.open_dataset(File)
       ds   = xr.decode_cf(ds)
       obs_t = ds["time"]
       obs_flx = ds["Vflx_tot"] * 1.e-6
       flx_avg_obs = np.nanmean(obs_flx)
       flx_std_obs = np.nanstd(obs_flx)

    else:

       obs = False
       # Values from Table 1 of 
       # Osterhus et al. 2019
       if sec == "latrabjarg": 
          flx_avg_obs = -3.2 
          flx_std_obs = 0.5
       if sec == "pos503-5": 
          flx_avg_obs = -0.4
          flx_std_obs = 0.3
       if sec == "ifr":
          flx_avg_obs = -0.4
          flx_std_obs = 0.3
       if sec == "fbc": 
          flx_avg_obs = -2.0
          flx_std_obs = 0.3
       if sec == "wtr": 
          flx_avg_obs = -0.2
          flx_std_obs = 0.1    
       if sec == "cgfz":
          flx_avg_obs = -1.7
          flx_std_obs = 0.5  

    print('Observations: ' + str(flx_avg_obs) + " +/- " + str(flx_std_obs))

    # zps --------------------------------------------------------------------
    File = mod_dir + '/zps/zps_' + sec + "_ovf_vflux.nc"
    ds   = xr.open_dataset(File)
    ds   = xr.decode_cf(ds)
    zps_t = ds["time"]
    zps_flx = ds["Vflx_tot"] * 1.e-6

    if obs:
       zps_flx_m = zps_flx.where(np.logical_and(zps_t.astype(datetime)>=obs_t.astype(datetime)[0], 
                                                zps_t.astype(datetime)<=obs_t.astype(datetime)[-1]))
    else:
       #zps_flx_m = zps_flx
       zps_flx_m = zps_flx.where(np.logical_and(zps_t>=np.datetime64(datetime(2014, 1, 1)),
                                                zps_t<np.datetime64(datetime(2020, 1, 1))))

    flx_avg_zps = np.nanmean(zps_flx_m)
    flx_std_zps = np.nanstd(zps_flx_m)
    print('Model zps: ' + str(flx_avg_zps) + " +/- " + str(flx_std_zps))
    print('rel error: ' + str(((flx_avg_zps-flx_avg_obs)/flx_avg_obs)*100.)+'%' )
    print('')

    # szt --------------------------------------------------------------------
    File = mod_dir + '/szt/szt_' + sec + "_ovf_vflux.nc"
    ds   = xr.open_dataset(File)
    szt_t = ds["time"]
    szt_flx = ds["Vflx_tot"] * 1.e-6

    if obs:
       szt_flx_m = szt_flx.where(np.logical_and(szt_t.astype(datetime)>=obs_t.astype(datetime)[0],
                                                szt_t.astype(datetime)<=obs_t.astype(datetime)[-1]))
    else:
       #szt_flx_m = szt_flx
       szt_flx_m = szt_flx.where(np.logical_and(szt_t>=np.datetime64(datetime(2014, 1, 1)),
                                                szt_t<np.datetime64(datetime(2020, 1, 1))))

    flx_avg_szt = np.nanmean(szt_flx_m)
    flx_std_szt = np.nanstd(szt_flx_m)
    print('Model szt: ' + str(flx_avg_szt)  + " +/- " + str(flx_std_szt))
    print('rel error: ' + str(((flx_avg_szt-flx_avg_obs)/flx_avg_obs)*100.)+'%' )
    print('')

    # MEs --------------------------------------------------------------------
    File = mod_dir + '/MEs/MEs_' + sec + "_ovf_vflux.nc" 
    ds   = xr.open_dataset(File)
    mes_t = ds["time"]
    mes_flx = ds["Vflx_tot"] * 1.e-6

    if obs:
       mes_flx_m = mes_flx.where(np.logical_and(mes_t.astype(datetime)>=obs_t.astype(datetime)[0],
                                                mes_t.astype(datetime)<=obs_t.astype(datetime)[-1]))
    else:
       #mes_flx_m = mes_flx
       mes_flx_m = mes_flx.where(np.logical_and(mes_t>=np.datetime64(datetime(2014, 1, 1)),
                                                mes_t<np.datetime64(datetime(2020, 1, 1))))

    flx_avg_mes = np.nanmean(mes_flx_m)
    flx_std_mes = np.nanstd(mes_flx_m)
    print('Model MEs: ' + str(flx_avg_mes) + " +/- " + str(flx_std_mes))
    print('rel error: ' + str(((flx_avg_mes-flx_avg_obs)/flx_avg_obs)*100.)+'%' )
    print('')

    if not obs:
       obs_t   = mes_t
       obs_flx = (mes_flx_m * 0.) + flx_avg_obs

#======================================================================================

    gs = GridSpec(1, 12, figure=fig)
    ax1 = fig.add_subplot(gs[0,:-2])
    ax2 = fig.add_subplot(gs[0,-2::],sharey=ax1)

    line = []
    Label = []

    # flx_tot
    l = ax1.plot(obs_t, obs_flx, linestyle="-", linewidth=6.5, color=cols[0])
    line.append(l[0])
    Label.append('observation')
    l = ax1.plot(zps_t, zps_flx, linestyle="-", linewidth=5.5, color=cols[1])
    line.append(l[0])
    Label.append('zps')
    l = ax1.plot(szt_t, szt_flx, linestyle="-", linewidth=5.5, color=cols[2])
    line.append(l[0])
    Label.append('szt')
    l = ax1.plot(mes_t, mes_flx, linestyle="-", linewidth=5.5, color=cols[3])
    line.append(l[0])
    Label.append('MEs')

    ax2.errorbar([1], [flx_avg_obs], yerr=flx_std_obs, fmt='-o', ms=20, color=cols[0],
                 ecolor='black', elinewidth=3, capsize=0 )
    ax2.errorbar([2], [flx_avg_zps], yerr=flx_std_zps, fmt='-o', ms=20, color=cols[1],
                 ecolor='black', elinewidth=3, capsize=0 )
    ax2.errorbar([3], [flx_avg_szt], yerr=flx_std_szt, fmt='-o', ms=20, color=cols[2],
                 ecolor='black', elinewidth=3, capsize=0 )
    ax2.errorbar([4], [flx_avg_mes], yerr=flx_std_mes, fmt='-o', ms=20, color=cols[3],
                 ecolor='black', elinewidth=3, capsize=0 )

    # ====================================================================
    # COMMON

    every_nth = 2

    ax1.set_ylabel(r'Transport [Sv]', fontsize='40',color="black")
    ax1.get_yaxis().set_label_coords(-0.08,0.5)
    ax1.tick_params(axis='y', labelsize=35)
    ax1.tick_params(axis='y', which='major', width=1.50, length=10)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.tick_params(axis='x', which='minor', width=2.00, length=10, labelsize=35)
    ax1.tick_params(axis='x', which='major', width=4.00, length=20, labelsize=35)
    #anchored_text = AnchoredText('(c)', loc=2, frameon=False, prop=dict(fontweight="bold",fontsize=30))
    #ax1.add_artist(anchored_text)

    ax2.get_yaxis().set_label_coords(-0.08,0.5)
    ax2.tick_params(axis='y', labelsize=35)
    ax2.tick_params(axis='y', which='major', width=1.50, length=10)
    
    ax1.set_xlim(datetime(2009, 11, 1), datetime(2019, 2, 1))
    ax2.set_xlim(-0.5,5)

    for n, label in enumerate(ax2.xaxis.get_minorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax2.xaxis.get_majorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax2.yaxis.get_minorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax2.yaxis.get_majorticklabels()): label.set_visible(False)


    ax1.grid(True)
    ax2.grid(True)    

    #plt.rc('legend', **{'fontsize':40})
    #print(tuple(line), Label)
    #fig.legend(tuple(line),Label,'upper center',ncol=4)
    figname = sec + '_timeseries.png'
    plt.subplots_adjust(hspace=0.15)
    plt.savefig(figname,bbox_inches="tight", pad_inches=0.1)
    plt.close()
