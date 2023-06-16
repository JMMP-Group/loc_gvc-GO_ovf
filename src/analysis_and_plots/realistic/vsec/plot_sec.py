#!/usr/bin/env python

import nsv
import numpy as np
import xarray as xr
from lib_sec import mpl_sec_loop
from utils import compute_masks, getArgs, extract_obs, extract_model
import matplotlib.colors as colors

def plot_setting(section):

    setting = {
     'latrabjarg_climatology':{ "ind": [0, -1],
                                "xlm": [0., 225.], 
                                "ylm": [0.,700.],
                                "Tcb": [-0.25,0.,0.25,0.5,0.75,1.,1.5,2.,
                                         2.5,3.,3.5,4.,4.5,5.,5.5,6.,7.,8.],
                                "Scb": [33.6,33.8,34.,34.2,34.4,34.5,34.6,34.7,34.75,34.8,
                                        34.85,34.9,34.92,34.96,34.98,35.,35.05],
                                "Dcb": [27.0,27.25,27.5,27.6,27.7,27.8,27.84,27.88,
                                        27.92,27.96,28.00]
                              },
     'kogur':{ "ind": [0, -1],
               "xlm": 'maxmin', 
               "ylm": [0.,1550.],
               "Tcb": [-1.,-0.5,-0.25,0.,0.25,0.5,0.75,1.,1.5,2.,2.5,3.,4.,5.,6.],
               "Scb": [33.6,33.8,34.0,34.2,34.4,34.6,34.7,34.8,34.825,34.85,
                       34.875,34.9,34.925,34.95],
               "Dcb": [27.0,27.25,27.5,27.6,27.7,27.8,27.84,27.88,
                       27.92,27.96,28.00]
             },
     'ovide':{ "ind": [0,37,105,110,142,365,450,462,-1],
               "xlm": [0., 2000.],
               "ylm": [0., 4000.],
               "Tcb": [2.,2.2,2.4,2.6,2.8,3.,3.2,3.4,3.6,3.8,4.0,4.5,5.,6.,7.,8.,9.,10.],
               "Scb": np.arange(34.90,35.07,0.01),
               "Dcb": [27.0,27.25,27.5,27.6,27.7,27.8,27.84,27.88,
                       27.92,27.96]
             },
     'osnap-seasonal':{ "ind": range(80,256), #[80,145,158,213,223,225,238,-1],
                        "xlm": [0., 1600.],
                        "ylm": [0., 3500.],
#                        "Tcb": [2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.5,4.,5.,6.,7.,8.,10.],
#                        "Scb": np.arange(34.88,35.05,0.01)
                        "Tcb": [2.,2.2,2.4,2.6,2.8,3.,3.2,3.4,3.6,3.8,4.0,4.5,5.,6.,7.,8.,9.,10.],
                        "Scb": np.arange(34.90,35.07,0.01),
                        "Dcb": [27.0,27.25,27.5,27.6,27.7,27.8,27.84,27.88,
                                27.92,27.96]
                      },
     'osnap-annual':{ "ind": range(80,256), #[80,145,158,213,223,225,238,-1],
                      "xlm": [0., 1800.],
                      "ylm": [0., 3500.],
#                      "Tcb": [2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.5,4.,5.,6.,7.,8.,10.],
#                      "Scb": np.arange(34.88,35.05,0.01)
                      "Tcb": [2.,2.2,2.4,2.6,2.8,3.,3.2,3.3,3.6,3.8,4.0,4.5,5.,6.,7.,8.,9.,10.],
                      "Scb": np.arange(34.90,35.07,0.01),
                      "Dcb": [27.0,27.25,27.5,27.6,27.7,27.8,27.84,27.88,
                              27.92,27.96]
                    },
     'ho2000':{ "ind": [0,7,-1],
                "xlm": "maxmin",
                "ylm": [0., 1100.],
                "Tcb": [-1.,-0.5,0.,0.5,1.,1.5,2.,2.2,2.4,2.6,2.8,3.,3.5,4.,5.,6.,8.,10.],
                "Scb": np.arange(34.90,35.4,0.03),
                "Dcb": [27.0,27.25,27.5,27.6,27.7,27.8,27.84,27.88,
                        27.92,27.96,28.]
              },
     'eel':{ "ind": None,
             "xlm": "maxmin",
             "ylm": [0., 2700.], #[0., 2685.],
             "Tcb": [2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.5,4.,4.5,5.,6.,7.,8.,10.,12.],
             #"Scb": np.arange(34.88,35.48,0.05),
             "Scb": [34.90,34.94,34.98,35.00,35.03,35.08,35.18,35.28,35.38,35.48],
             "Dcb": [27.0,27.25,27.5,27.6,27.7,27.8,27.84,27.88,
                     27.92,27.96]
           },
     'pos503-5':{ "ind": None,
                  "xlm": "maxmin",
                  "ylm": [0., 500.],
                  "Tcb": [0.,0.25,0.5,0.75,1.,1.5,2.,2.2,2.4,2.6,2.8,3.,3.5,4.,5.,6.,8.,10.],
                  "Scb": np.arange(34.9,35.4,0.03),
                  "Dcb": [27.0,27.25,27.5,27.6,27.7,27.8,27.84,27.88,
                          27.92,27.96,28.0]
           },
     'm82_1-4':{"ind": [-1,-10,0],
                "xlm": "maxmin",
                "ylm": [0., 1020.],
                "Tcb": [-1.,-0.5,0.,0.5,1.,1.5,2.,2.2,2.4,2.6,2.8,3.,3.5,4.,5.,6.,8.],
                "Scb": np.arange(34.90,35.4,0.03),
                "Dcb": [27.0,27.25,27.5,27.6,27.7,27.8,27.84,27.88,
                        27.92,27.96,28.]               
               },
     'm82_1-6':{"ind": [-1, 0],
                "xlm": "maxmin",
                "ylm": [0., 1600.],
                "Tcb": [-1.,-0.5,0.,0.5,1.,1.5,2.,2.2,2.4,2.6,2.8,3.,3.5,4.,5.,6.,8.],
                "Scb": np.arange(34.90,35.4,0.03),
                "Dcb": [27.0,27.25,27.5,27.6,27.7,27.8,27.84,27.88,
                        27.92,27.96,28.]  
               },
     'm82_1-7':{"ind": [0, -1],
                "xlm": "maxmin",
                "ylm": [0., 2200.],
                "Tcb": [-1.,-0.5,0.,0.5,1.,1.5,2.,2.2,2.4,2.6,2.8,3.,3.5,4.,5.,6.,8.],
                "Scb": np.arange(34.90,35.4,0.03),
                "Dcb": [27.0,27.25,27.5,27.6,27.7,27.8,27.84,27.88,
                        27.92,27.96,28.] 
               },
     'm82_1-8':{"ind": [0, -1],
                "xlm": "maxmin",
                "ylm": [0., 2150.],
                "Tcb": [-1.,-0.5,0.,0.5,1.,1.5,2.,2.2,2.4,2.6,2.8,3.,3.5,4.,5.,6.,8.],
                "Scb": np.arange(34.90,35.4,0.03),
                "Dcb": [27.0,27.25,27.5,27.6,27.7,27.8,27.84,27.88,
                        27.92,27.96,28.] 
               },
    }

    return setting[section]

if __name__ == "__main__":

   args = getArgs()
   # Read in observations
   if "m82_1" in str(args.sec[0]):
      section = args.sec[0][-1]
      obs_nsv = "nsv.Standardizer().m82_1(" + section  + ")"
   elif "kn203_2" in str(args.sec[0]):
      section = args.sec[0][-1]
      obs_nsv = 'nsv.Standardizer().kn203_2("' + section  + '")'
   elif "pos503" in str(args.sec[0]):
      section = args.sec[0][-1]
      obs_nsv = 'nsv.Standardizer().pos503(' + section  + ')'
   elif "osnap" in str(args.sec[0]):
      obs_nsv = 'nsv.Standardizer().osnap'
   else:
      obs_nsv = "nsv.Standardizer()."+args.sec[0]
   ds_obs = eval(obs_nsv)
   if "osnap" in str(args.sec[0]): ds_obs.attrs['description'] = str(args.sec[0]).upper()
   ds_obs = ds_obs.drop('station_id')  
 
   # Preparing arrays for plotting
   msk_mes = None
   if args.obs:              # OBS
      ds_T = extract_obs(ds_obs)
      lon3 = ds_T.longitude.isel(t=0).values 
      lat3 = ds_T.latitude.isel(t=0).values
      tdep = ds_T.depth.isel(t=0).values
      wdep = ds_T.depth.isel(t=0).values
      msk3 = ds_T.mask.isel(t=0).values
      bat2 = ds_T.sea_floor_depth_below_geoid.values
   else:                     # MODEL
      xmin = 880
      xmax = 1150
      ymin = 880
      ymax = 1140
      model_file = (args.zps[0] if args.zps else args.mes[0] if args.mes else args.szt[0])
      ds_T = extract_model(args.domcfg[0], model_file, xmin, xmax, ymin, ymax)
      if args.mes or args.szt:
         ds_msk = xr.open_dataset(args.locmsk[0])
         ds_msk = ds_msk.isel(x=slice(xmin, xmax),
                              y=slice(ymin, ymax))
         if "s2z_msk" in ds_msk.variables:
            msk_mes = ds_msk["s2z_msk"].values
            msk_mes[msk_mes>0] = 1
         del ds_msk
      lon3 = ds_T.glamt.values
      lat3 = ds_T.gphit.values
      tdep = ds_T.gdept_0.values
      wdep = ds_T.gdepw_0.values
      msk3 = ds_T.tmask.values
      bat2 = ds_T.bathymetry.values

   tem4 = ds_T.potential_temperature.values
   sal4 = ds_T.practical_salinity.values
   rho4 = ds_T.sigma_theta.values

   # Plotting

   CMAP_sal = colors.ListedColormap(['purple','indigo','darkblue', 
                                 'dodgerblue','deepskyblue','lightskyblue',
#                                 'mediumspringgreen','lime','greenyellow','yellow','gold','orange',
                                 'lime','greenyellow','yellow','gold','orange',
                                 'darkorange','orangered','red','firebrick','darkred','gray'])

   CMAP_tem = colors.ListedColormap(['lightgray','darkgray','dimgray',
                                     'lightskyblue','deepskyblue','dodgerblue','steelblue',
                                     'forestgreen','limegreen','lime','greenyellow',
                                     'yellow','gold','orange',
                                     'orangered','firebrick','indigo'])

   # Common setting
   
   st = plot_setting(args.sec[0])

   sec_lon = ds_obs.longitude
   sec_lat = ds_obs.latitude

   if 'osnap' in str(args.sec[0]):
      if args.obs:
         sec_lon = sec_lon[80::]
         sec_lat = sec_lat[80::]

   if 'eel' in str(args.sec[0]):
       #print(sec_lon.where(sec_lon>-13.4).dropna("station"))
       #print(sec_lat.where(sec_lon>-13.4).dropna("station"))
       #quit() 
      sec_lon = sec_lon.where(sec_lon<-13.4) # Icelandic Basin
      sec_lat = sec_lat.where(sec_lon<-13.4)

   #if not ('osnap' in str(args.sec[0]) and not args.obs):
   sec_lon = sec_lon.values.tolist()
   sec_lat = sec_lat.values.tolist()

   sec_lon = [x for x in sec_lon if str(x) != 'nan']
   sec_lat = [x for x in sec_lat if str(x) != 'nan']

   if args.obs:
      sec_I = [sec_lon]
      sec_J = [sec_lat]
   else:
      if st['ind'] is not None:
         sec_I     = [list(sec_lon[i] for i in st['ind'])]
         sec_J     = [list(sec_lat[i] for i in st['ind'])]
      else:
         sec_I = [sec_lon]
         sec_J = [sec_lat]
   xcoord    = 'dist'
   xlim      = st['xlm']
   ylim      = st['ylm']
   vlevel    = ('no' if args.obs else 'Z_ps' if args.zps else 'MEs' if args.mes else 'SZT')
   first_zlv = (49 if args.szt else None)
   xgrid     = "false"
   rbat      = ("true" if args.obs else "false")
   mbat      = "true"
   mbat_ln   = "false"
   check     = 'false'
   check_val = 'false'
   cn_line   = "false"

   proj = []

   if "m82_1" in str(args.sec[0]):
      figwidth  = 17.5
      figheight = 34.
   elif 'osnap' in str(args.sec[0]):
      figwidth  = 26.
      figheight = 17.5
   elif "latrabjarg" in str(args.sec[0]):
      figwidth  = 24.
      figheight = 17.5
   elif "pos503" in str(args.sec[0]):
      figwidth  = 26.
      figheight = 17.5
   else:
      figwidth  = 22.
      figheight = 17.5

   if 'osnap' in str(args.sec[0]):
      sec = 'osnap'
   elif "latrabjarg" in str(args.sec[0]):
      sec = "Latr" 
   elif "m82_1" in str(args.sec[0]):
      sec = "m82_1-8"
   elif "eel" in str(args.sec[0]):
      sec = "eel"
   else:
      sec = "dummy"

   # TEMPERATURE
   var_strng  = "Temperature"
   unit_strng = "[$^\circ$C]"
   date       = ""
   timeres_dm = "1d"
   timestep   = (range(len(ds_T.t)) if "t" in ds_T else [0])
   PlotType   = "contourf"
   colmap     = CMAP_tem
   cn_level   = st['Tcb'] 
   varlim     = [cn_level[0], cn_level[-1]]

   mpl_sec_loop('Temperature', '.png', var_strng, unit_strng, 
                date, timeres_dm, timestep, PlotType,
                sec_I, sec_J, lon3, lat3, tdep, wdep, msk3, tem4, proj,
                xcoord, vlevel, bat2, [], rbat, mbat_ln, mbat,
                xlim, ylim, varlim, check, check_val, xgrid, 
                colmap=colmap, cn_level=cn_level, cn_line=cn_line, 
                var_aux=rho4, var_aux_lev=st['Dcb'], msk_mes=msk_mes, first_zlv=first_zlv,
                figwidth=figwidth,figheight=figheight,sec=sec)

   # SALINITY
   var_strng  = "Salinity"
   unit_strng = "[PSU]"
   date       = ""
   timeres_dm = "1d"
   timestep   = range(len(ds_T.t))
   PlotType   = "contourf"
   colmap     = CMAP_sal
   cn_level   = st['Scb']
   varlim     = [cn_level[0], cn_level[-1]]

   mpl_sec_loop('Salinity', '.png', var_strng, unit_strng,
                date, timeres_dm, timestep, PlotType,
                sec_I, sec_J, lon3, lat3, tdep, wdep, msk3, sal4, proj,
                xcoord, vlevel, bat2, [], rbat, mbat_ln, mbat,
                xlim, ylim, varlim, check, check_val, xgrid,
                colmap=colmap, cn_level=cn_level, cn_line=cn_line, 
                var_aux=rho4, var_aux_lev=st['Dcb'], msk_mes=msk_mes, first_zlv=first_zlv,
                figwidth=figwidth,figheight=figheight,sec=sec)

   # POTENTIAL DENSITY
   var_strng  = "Pot_Density"
   unit_strng = "[$kg\;m^{-3}$]"
   date       = ""
   timeres_dm = "1d"
   timestep   = range(len(ds_T.t))
   PlotType   = "contourf"
   colmap     = "viridis"
   cn_level   = st['Dcb']
   varlim     = [cn_level[0], cn_level[-1]]

   mpl_sec_loop('Pot_Density', '.png', var_strng, unit_strng,
                date, timeres_dm, timestep, PlotType,
                sec_I, sec_J, lon3, lat3, tdep, wdep, msk3, rho4, proj,
                xcoord, vlevel, bat2, [], rbat, mbat_ln, mbat,
                xlim, ylim, varlim, check, check_val, xgrid,
                colmap=colmap, cn_level=cn_level, cn_line=cn_line,
                var_aux=rho4, var_aux_lev=st['Dcb'], msk_mes=msk_mes, first_zlv=first_zlv,
                figwidth=figwidth,figheight=figheight,sec=sec)

