#!/usr/bin/env python

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as mtick
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import AnchoredText
from matplotlib.gridspec import GridSpec

# ==============================================================================
# Input parameters

# 1. INPUT FILES
# Change this to match your local paths set-up
base_dir = "/your_local_path"

data_dir = base_dir + '/realistic'

sections = ["ho2000","latrabjarg","osnap-IB","osnap-IS","pos503-5"]
cols = ["black","red","dodgerblue","limegreen"]

for sec in sections:

    print("====", sec)

    fig = plt.figure(figsize = (22.,17.5), dpi=100, constrained_layout=True)

    # OBS
    File = data_dir + "/obs/ovf_TS/" sec + "_ovf_properties.nc"
    ds   = xr.open_dataset(File)
    if sec == "ho2000" or sec == "latrabjarg" or sec == "pos503-5":
       obs_t = np.asarray([datetime(2014, 1, 1), datetime(2018, 12, 31)])
       obs_rho = [ds["rho_avg"].values[0], ds["rho_avg"].values[0]]
       obs_tem = [ds["tem_avg"].values[0], ds["tem_avg"].values[0]]
       obs_sal = [ds["sal_avg"].values[0], ds["sal_avg"].values[0]]
       rho_avg_obs = ds["rho_avg"].values[0]
       rho_std_obs = 0.
       tem_avg_obs = ds["tem_avg"].values[0]
       tem_std_obs = 0.
       sal_avg_obs = ds["sal_avg"].values[0]
       sal_std_obs = 0.
    else:
       obs_t = ds["time"]
       obs_rho = ds["rho_avg"]
       obs_tem = ds["tem_avg"]
       obs_sal = ds["sal_avg"]
       rho_avg_obs = np.nanmean(obs_rho)
       rho_std_obs = np.nanstd(obs_rho)
       tem_avg_obs = np.nanmean(obs_tem)
       tem_std_obs = np.nanstd(obs_tem)
       sal_avg_obs = np.nanmean(obs_sal)
       sal_std_obs = np.nanstd(obs_sal)

    print('Observations:') 
    print('   1) rho: ' + str(rho_avg_obs) + " +/- " + str(rho_std_obs))
    print('   2) tem: ' + str(tem_avg_obs) + " +/- " + str(tem_std_obs))
    print('   3) sal: ' + str(sal_avg_obs) + " +/- " + str(sal_std_obs))

    # zps
    File = data_dir + '/zps/zps_' + sec + "_ovf_properties.nc"
    ds   = xr.open_dataset(File)
    ds   = xr.decode_cf(ds)
    zps_t = ds["time"].astype("datetime64[ns]")
    zps_rho = ds["rho_avg"]
    zps_tem = ds["tem_avg"]
    zps_sal = ds["sal_avg"]

    if sec == "ho2000" or sec == "latrabjarg" or sec == "pos503-5":
       zps_rho_m = zps_rho
       zps_tem_m = zps_tem
       zps_sal_m = zps_sal
    else:
       zps_rho_m = zps_rho.where(np.logical_and(zps_t>=obs_t[0],
                                                zps_t<=obs_t[-1]))
       zps_tem_m = zps_tem.where(np.logical_and(zps_t>=obs_t[0],
                                                zps_t<=obs_t[-1]))
       zps_sal_m = zps_sal.where(np.logical_and(zps_t>=obs_t[0],
                                                zps_t<=obs_t[-1]))

    rho_avg_zps = np.nanmean(zps_rho_m)
    rho_std_zps = np.nanstd(zps_rho_m)
    tem_avg_zps = np.nanmean(zps_tem_m)
    tem_std_zps = np.nanstd(zps_tem_m)
    sal_avg_zps = np.nanmean(zps_sal_m)
    sal_std_zps = np.nanstd(zps_sal_m)

    print('Model zps:')
    print('   1) rho: ' + str(rho_avg_zps) + " +/- " + str(rho_std_zps))
    print('   2) tem: ' + str(tem_avg_zps) + " +/- " + str(tem_std_zps))
    print('   3) sal: ' + str(sal_avg_zps) + " +/- " + str(sal_std_zps))

    # szt
    File = data_dir + '/szt/szt_' + sec + "_ovf_properties.nc"
    ds   = xr.open_dataset(File)
    szt_t = ds["time"].astype("datetime64[ns]")
    szt_rho = ds["rho_avg"]
    szt_tem = ds["tem_avg"]
    szt_sal = ds["sal_avg"]

    if sec == "ho2000" or sec == "latrabjarg" or sec == "pos503-5":
       #szt_rho_m = szt_rho
       #szt_tem_m = szt_tem
       #szt_sal_m = szt_sal
       szt_rho_m = szt_rho.where(np.logical_and(szt_t>=np.datetime64(datetime(2014, 1, 1)),
                                                szt_t<=np.datetime64(datetime(2019, 1, 1))))
       szt_tem_m = szt_tem.where(np.logical_and(szt_t>=np.datetime64(datetime(2014, 1, 1)),
                                                szt_t<=np.datetime64(datetime(2019, 1, 1))))
       szt_sal_m = szt_sal.where(np.logical_and(szt_t>=np.datetime64(datetime(2014, 1, 1)),
                                                szt_t<=np.datetime64(datetime(2019, 1, 1))))
    else:
       szt_rho_m = szt_rho.where(np.logical_and(szt_t>=obs_t[0],
                                            szt_t<=obs_t[-1]))
       szt_tem_m = szt_tem.where(np.logical_and(szt_t>=obs_t[0],
                                            szt_t<=obs_t[-1]))
       szt_sal_m = szt_sal.where(np.logical_and(szt_t>=obs_t[0],
                                            szt_t<=obs_t[-1]))

    rho_avg_szt = np.nanmean(szt_rho_m)
    rho_std_szt = np.nanstd(szt_rho_m)
    tem_avg_szt = np.nanmean(szt_tem_m)
    tem_std_szt = np.nanstd(szt_tem_m)
    sal_avg_szt = np.nanmean(szt_sal_m)
    sal_std_szt = np.nanstd(szt_sal_m)

    print('Model szt:')
    print('   1) rho: ' + str(rho_avg_szt) + " +/- " + str(rho_std_szt))
    print('   2) tem: ' + str(tem_avg_szt) + " +/- " + str(tem_std_szt))
    print('   3) sal: ' + str(sal_avg_szt) + " +/- " + str(sal_std_szt))

    # MEs
    File = data_dir + '/MEs/MEs_' + sec + "_ovf_properties.nc" 
    ds   = xr.open_dataset(File)
    mes_t = ds["time"].astype("datetime64[ns]")
    mes_rho = ds["rho_avg"]
    mes_tem = ds["tem_avg"]
    mes_sal = ds["sal_avg"]

    if sec == "ho2000" or sec == "latrabjarg" or sec == "pos503-5":
       mes_rho_m = mes_rho
       mes_tem_m = mes_tem
       mes_sal_m = mes_sal
    else:
       mes_rho_m = mes_rho.where(np.logical_and(mes_t>=obs_t[0],
                                                mes_t<=obs_t[-1]))
       mes_tem_m = mes_tem.where(np.logical_and(mes_t>=obs_t[0],
                                                mes_t<=obs_t[-1]))
       mes_sal_m = mes_sal.where(np.logical_and(mes_t>=obs_t[0],
                                                mes_t<=obs_t[-1]))

    rho_avg_mes = np.nanmean(mes_rho_m)
    rho_std_mes = np.nanstd(mes_rho_m)
    tem_avg_mes = np.nanmean(mes_tem_m)
    tem_std_mes = np.nanstd(mes_tem_m)
    sal_avg_mes = np.nanmean(mes_sal_m)
    sal_std_mes = np.nanstd(mes_sal_m)

    print('Model MEs:')
    print('   1) rho: ' + str(rho_avg_mes) + " +/- " + str(rho_std_mes))
    print('   2) tem: ' + str(tem_avg_mes) + " +/- " + str(tem_std_mes))
    print('   3) sal: ' + str(sal_avg_mes) + " +/- " + str(sal_std_mes))


#======================================================================================

    gs = GridSpec(3, 12, figure=fig)
    
    ax1 = fig.add_subplot(gs[0,:-2])
    ax2 = fig.add_subplot(gs[1,:-2],sharex=ax1)
    ax3 = fig.add_subplot(gs[2,:-2],sharex=ax1)
    ax4 = fig.add_subplot(gs[0,-2::], sharey=ax1)
    ax5 = fig.add_subplot(gs[1,-2::],sharex=ax4, sharey=ax2)
    ax6 = fig.add_subplot(gs[2,-2::],sharex=ax4, sharey=ax3)

    line = []
    Label = []

    # rho_avg
    l = ax1.plot(obs_t, obs_rho, linestyle="-", linewidth=6.5, color=cols[0])
    line.append(l[0])
    Label.append('observation')
    l = ax1.plot(zps_t, zps_rho, linestyle="-", linewidth=5.5, color=cols[1])
    line.append(l[0])
    Label.append('zps')
    l = ax1.plot(szt_t, szt_rho, linestyle="-", linewidth=5.5, color=cols[2])
    line.append(l[0])
    Label.append('szt')
    l = ax1.plot(mes_t, mes_rho, linestyle="-", linewidth=5.5, color=cols[3])
    line.append(l[0])
    Label.append('MEs')

    ax4.errorbar([1], [rho_avg_obs], yerr=rho_std_obs, fmt='-o', ms=15, color=cols[0],
                 ecolor='black', elinewidth=3, capsize=0 )
    ax4.errorbar([2], [rho_avg_zps], yerr=rho_std_zps, fmt='-o', ms=15, color=cols[1],
                 ecolor='black', elinewidth=3, capsize=0 )
    ax4.errorbar([3], [rho_avg_szt], yerr=rho_std_szt, fmt='-o', ms=15, color=cols[2],
                 ecolor='black', elinewidth=3, capsize=0 )
    ax4.errorbar([4], [rho_avg_mes], yerr=rho_std_mes, fmt='-o', ms=15, color=cols[3],
                 ecolor='black', elinewidth=3, capsize=0 )

    # tem_avg
    l = ax2.plot(obs_t, obs_tem, linestyle="-", linewidth=6.5, color=cols[0])
    l = ax2.plot(zps_t, zps_tem, linestyle="-", linewidth=5.5, color=cols[1])
    l = ax2.plot(szt_t, szt_tem, linestyle="-", linewidth=5.5, color=cols[2])
    l = ax2.plot(mes_t, mes_tem, linestyle="-", linewidth=5.5, color=cols[3])

    ax5.errorbar([1], [tem_avg_obs], yerr=tem_std_obs, fmt='o', ms=15, color=cols[0],
                 ecolor='black', elinewidth=3, capsize=0 )
    ax5.errorbar([2], [tem_avg_zps], yerr=tem_std_zps, fmt='o', ms=15, color=cols[1],
                 ecolor='black', elinewidth=3, capsize=0 )
    ax5.errorbar([3], [tem_avg_szt], yerr=tem_std_szt, fmt='o', ms=15, color=cols[2],
                 ecolor='black', elinewidth=3, capsize=0 )
    ax5.errorbar([4], [tem_avg_mes], yerr=tem_std_mes, fmt='o', ms=15, color=cols[3],
                 ecolor='black', elinewidth=3, capsize=0 )

    # sal_avg
    l = ax3.plot(obs_t, obs_sal, linestyle="-", linewidth=6.5, color=cols[0])
    l = ax3.plot(zps_t, zps_sal, linestyle="-", linewidth=5.5, color=cols[1])
    l = ax3.plot(szt_t, szt_sal, linestyle="-", linewidth=5.5, color=cols[2])
    l = ax3.plot(mes_t, mes_sal, linestyle="-", linewidth=5.5, color=cols[3])

    ax6.errorbar([1], [sal_avg_obs], yerr=sal_std_obs, fmt='o', ms=15, color=cols[0],
                 ecolor='black', elinewidth=3, capsize=0 )
    ax6.errorbar([2], [sal_avg_zps], yerr=sal_std_zps, fmt='o', ms=15, color=cols[1],
                 ecolor='black', elinewidth=3, capsize=0 ) 
    ax6.errorbar([3], [sal_avg_szt], yerr=sal_std_szt, fmt='o', ms=15, color=cols[2],
                 ecolor='black', elinewidth=3, capsize=0 ) 
    ax6.errorbar([4], [sal_avg_mes], yerr=sal_std_mes, fmt='o', ms=15, color=cols[3],
                 ecolor='black', elinewidth=3, capsize=0 )

    # ====================================================================
    # COMMON

    every_nth = 2

    ax1.set_ylabel(r'$\sigma_{\theta}$ [$kg\;m^{-3}$]', fontsize='27',color="black")
    ax1.get_yaxis().set_label_coords(-0.08,0.5)
    ax1.tick_params(axis='y', labelsize=27)
    ax1.tick_params(axis='y', which='major', width=1.50, length=10)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #anchored_text = AnchoredText('(a)', loc=2, frameon=False, prop=dict(fontweight="bold",fontsize=30))
    #ax1.add_artist(anchored_text)

    ax2.set_ylabel('T [$^{\circ}$C]', fontsize='27',color="black")
    ax2.get_yaxis().set_label_coords(-0.08,0.5)
    ax2.tick_params(axis='y', labelsize=27)
    ax2.tick_params(axis='y', which='major', width=1.50, length=10)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #anchored_text = AnchoredText('(b)', loc=2, frameon=False, prop=dict(fontweight="bold",fontsize=30))
    #ax2.add_artist(anchored_text)

    ax3.set_ylabel('S [$PSU$]', fontsize='27',color="black")
    ax3.get_yaxis().set_label_coords(-0.08,0.5)
    ax3.tick_params(axis='y', labelsize=27)
    ax3.tick_params(axis='y', which='major', width=1.50, length=10)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.tick_params(axis='x', which='minor', width=2.00, length=10, labelsize=35)
    ax3.tick_params(axis='x', which='major', width=4.00, length=20, labelsize=35)
    #anchored_text = AnchoredText('(c)', loc=2, frameon=False, prop=dict(fontweight="bold",fontsize=30))
    #ax3.add_artist(anchored_text)

    ax3.set_xlim(datetime(2009, 11, 1), datetime(2019, 2, 1))

    ax4.get_yaxis().set_label_coords(-0.08,0.5)
    ax4.tick_params(axis='y', labelsize=27)
    ax4.tick_params(axis='y', which='major', width=1.50, length=10)
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax4.set_xlim(-0.5,5)

    ax5.get_yaxis().set_label_coords(-0.08,0.5)
    ax5.tick_params(axis='y', labelsize=27)
    ax5.tick_params(axis='y', which='major', width=1.50, length=10)
    ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax6.get_yaxis().set_label_coords(-0.08,0.5)
    ax6.tick_params(axis='y', labelsize=27)
    ax6.tick_params(axis='y', which='major', width=1.50, length=10)
    ax6.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    for n, label in enumerate(ax1.xaxis.get_minorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax1.xaxis.get_majorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax2.xaxis.get_minorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax2.xaxis.get_majorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax4.xaxis.get_minorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax4.xaxis.get_majorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax5.xaxis.get_minorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax5.xaxis.get_majorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax6.xaxis.get_minorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax6.xaxis.get_majorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax4.yaxis.get_minorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax4.yaxis.get_majorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax5.yaxis.get_minorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax5.yaxis.get_majorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax6.yaxis.get_minorticklabels()): label.set_visible(False)
    for n, label in enumerate(ax6.yaxis.get_majorticklabels()): label.set_visible(False)


    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    ax6.grid(True)
    plt.rc('legend', **{'fontsize':30})
    #print(tuple(line), Label)
    #fig.legend(tuple(line),Label, ncol=2)#,'upper center',ncol=4)
    figname = sec + '_timeseries.png'
    plt.subplots_adjust(hspace=0.15)
    plt.savefig(figname,bbox_inches="tight", pad_inches=0.1)
    plt.close()
