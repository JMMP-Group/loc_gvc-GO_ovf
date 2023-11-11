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

def comp_pot_sigma(depth, practical_salinity, potential_temperature):
    # i=1015, j=977
    latitude  =  60.3004
    longitude = -34.3571
    # Absolute Salinity
    press = gsw.p_from_z(-depth, latitude)
    abs_s = gsw.SA_from_SP(practical_salinity,
                           press,
                           longitude,
                           latitude
    )
    # Conservative Temperature
    con_t = gsw.CT_from_pt(abs_s,
                           potential_temperature
    )
    # Potential density anomaly
    sigma_theta = gsw.density.sigma0(abs_s, con_t) 

    return sigma_theta


# 1) OSNAP for ambient stratification

ds_osnap = nsv.Standardizer().osnap
ds_osnap = ds_osnap.mean("time")

ds_osnap = compute_potential_sigma(ds_osnap)
sigma = ds_osnap.sigma_theta

# Real profile from OSNAP
sigma_station_115 = sigma.isel(station=115).values
depth_station_115 = sigma.isel(station=115).depth.values

sigma_station_180 = sigma.isel(station=180).values
depth_station_180 = sigma.isel(station=180).depth.values

# 2) SYNTHETIC PROFILES  

# Depths

zdep = [    5.02,   15.08,   25.16,   35.28,   45.45,   55.69,
           66.04,   76.55,   87.27,   98.31,  109.81,  121.95,
          135.03,  149.43,  165.73,  184.70,  207.43,  235.39,
          270.53,  315.37,  372.97,  446.80,  540.50,  657.32,
          799.55,  968.00, 1161.81, 1378.66, 1615.29, 1868.07,
         2133.52, 2408.58, 2690.78, 2978.17, 3269.28, 3563.04,
         3858.68, 4155.63, 4453.50, 4752.02, 5050.99, 6300.27  ]

#  Tracers, T/S July off-shelf

ztem = [ 13.0669, 12.8587, 12.4760, 11.9986, 11.5363, 11.1627,
         10.8898, 10.6753, 10.4927, 10.3334, 10.2182, 10.1457,
         10.1038, 10.0734, 10.0389,  9.9968,  9.9459,  9.8836,
          9.8069,  9.6953,  9.5345,  9.2901,  8.9319,  8.4192,
          7.7006,  6.7895,  5.7774,  4.8576,  4.1510,  3.6716,
          3.3331,  3.0606,  2.8275,  2.6317,  2.4735,  2.3497,
          2.2601,  2.1973,  2.1555,  2.1237,  2.1072,  2.1000 ]

zsal = [ 35.2001, 35.2052, 35.2186, 35.2411, 35.2661, 35.2873,
         35.3021, 35.3124, 35.3205, 35.3267, 35.3304, 35.3330,
         35.3355, 35.3393, 35.3422, 35.3438, 35.3436, 35.3428,
         35.3413, 35.3374, 35.3313, 35.3239, 35.3192, 35.3171,
         35.3171, 35.3171, 35.3171, 35.3171, 35.3171, 35.3171,
         35.3171, 35.3171, 35.3171, 35.3171, 35.3171, 35.3171,
         35.3171, 35.3171, 35.3171, 35.3171, 35.3171, 35.3171  ]

zsig = comp_pot_sigma(np.asarray(zdep), np.asarray(zsal), np.asarray(ztem))

# 3) PLOTS

fig  = plt.figure(figsize=(6,10))
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
ax   = fig.add_subplot(spec[:1])

pc_115 = ax.plot(sigma_station_115, depth_station_115, 'blueviolet', linewidth=3)
pc_180 = ax.plot(sigma_station_180, depth_station_180, 'magenta', linewidth=3)
pc_syn = ax.plot(zsig, zdep, 'black', linewidth=5)

plt.gca().invert_yaxis()
ax.set_ylabel('Depth [$m$]', fontsize='25',color="black")
ax.set_xlabel(r'$\sigma_{\theta}$ [$kg\;m^{-3}$]', fontsize='25',color="black")
#ax.set_xlim(26.5,28.25)
ax.set_ylim(3200,0)
ax.tick_params(axis='y', labelsize=25)
ax.tick_params(axis='y', which='major', width=1.50, length=10)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='x', which='major', width=1.50, length=10)

ax.grid(True)
fig_name = 'density_ini_hpge_osnap_DS_IS.png'
plt.savefig(fig_name, bbox_inches="tight")
print("done")
plt.close()

