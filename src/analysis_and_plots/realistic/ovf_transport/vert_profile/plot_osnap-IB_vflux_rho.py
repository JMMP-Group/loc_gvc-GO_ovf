#!/usr/bin/env python

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

#=======================================================================================
# Input parameters

# Change this to match your local paths set-up
base_dir = "/your_local_path"

exp = [base_dir + "/realistic/obs/ovf_transport/vflux_rho-IB_obs.nc",
       base_dir + "/realistic/zps/vflux_rho-IB_zps.nc",
       base_dir + "/realistic/szt/vflux_rho-IB_szt.nc",
       base_dir + "/realistic/MEs/vflux_rho-IB_MEs.nc"
      ]

col = ["black","red","blue","limegreen"]
lab = ["obs","zps","szt","MEs"]

#cm = 1/2.54  # centimeters in inches
#fig = plt.figure(figsize=(13*cm, 20*cm), dpi=200)
fig = plt.figure(figsize=(15, 25), dpi=100)
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
ax = fig.add_subplot(spec[:1])

minflux = 0.
maxflux = 0.

for e in range(len(exp)):
    ds = xr.open_dataset(exp[e]).squeeze()
    time = "t"
    sigm = "rho_bins"
    flux = "vflux_rho"
    a = 0.2

    flux_mean = ds[flux].mean(dim=time).values
    flux_std = ds[flux].std(dim=time).values
    sigma = ds[sigm].values 
    flux_mean = flux_mean / 10**6
    if e > 0: flux_mean = -flux_mean
    minflux = np.minimum(minflux,np.nanmin(flux_mean))
    maxflux = np.maximum(maxflux,np.nanmax(flux_mean))
 
    ax.plot(flux_mean, sigma, color=col[e], linestyle="-", linewidth=6.0, label=lab[e])
    #ax.fill_betweenx(sigma, mocsig_mean+mocsig_std, mocsig_mean-mocsig_std, alpha=a, facecolor=col[e])
    
ax.plot([minflux, maxflux], [27.85, 27.85], color="magenta", linestyle="--", linewidth=3.5)
ax.plot([minflux, maxflux], [27.84, 27.84], color="k", linestyle="--", linewidth=3.5)
ax.plot(flux_mean*0.0, sigma, color='black', linestyle="-", linewidth=2.)

plt.rc('legend', **{'fontsize':40})
ax.legend(loc='upper left', ncol=1, frameon=False)
ax.tick_params(axis='y', labelsize=40)
ax.tick_params(axis='y', which='major', width=1.50, length=20)
ax.tick_params(axis='x', labelsize=40)
ax.tick_params(axis='x', which='major', width=1.50, length=20)
ax.set_xlabel('Vol. Transport [Sv]', fontsize='40')
ax.set_ylabel(r'$\sigma_{\theta}$ [$kg\;m^{-3}$]', fontsize='40')
ax.set_ylim(27.5, 28)
plt.gca().invert_yaxis()
name = 'osnap-IB_vflux_rho_no-damp.png'
plt.savefig(name, bbox_inches="tight", pad_inches=0.1)


