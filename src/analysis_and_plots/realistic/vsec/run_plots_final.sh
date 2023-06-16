#!/bin/bash

plot_obs=1

# Change this to match your local paths set-up
vc_root="/your_local_path"

dom_zps=${vc_root}"/loc_gvc-nordic_ovf/models_geometry/dom_cfg/realistic/zps/domain_cfg_zps.nc"
dom_MEs=${vc_root}"/loc_gvc-nordic_ovf/models_geometry/dom_cfg/realistic/mes/domain_cfg_mes.nc"
dom_szt=${vc_root}"/loc_gvc-nordic_ovf/models_geometry/dom_cfg/realistic/szt/domain_cfg_szt.nc"
msk_loc=${vc_root}"/loc_gvc-nordic_ovf/models_geometry/loc_area/bathymetry.loc_area.dep2800_novf_sig1_stn9_itr1.nc"

data_root=${vc_root}"/loc_gvc-nordic_ovf/outputs/realistic"
zps_outdir=${data_root}"/zps"
szt_outdir=${data_root}"/szt"
MEs_outdir=${data_root}"/MEs"

# OSNAP ------------------------------------------------------------------------------------------------

for section in {"osnap-annual","latrabjarg_climatology","pos503-5","ho2000"}; do 

    echo ${section}"... "

    # OBS
    if [ $plot_obs -eq 1 ]; then
       python plot_sec.py -obs ${section}

       if [ ${section} == "osnap-seasonal" ]; then

          for SSN in {"SON","DJF","MAM","JJA"}; do
              case $SSN in
                   "DJF")
                        n=1
                        ;;
                   "JJA")
                        n=2
                        ;;
                   "MAM")
                        n=3
                        ;;
                   "SON")
                        n=4
                        ;;
              esac
              outfile=`ls sec_Temperature*_contourf_maxdepth_*${n}.png`
              mv $outfile "obs-"${section}"_TEM_sec_${SSN}.png"
              outfile=`ls sec_Salinity*_contourf_maxdepth_*${n}.png`
              mv $outfile "obs-"${section}"_SAL_sec_${SSN}.png"
              outfile=`ls sec_Pot_Density*_contourf_maxdepth_*${n}.png`
              mv $outfile "obs-"${section}"_DEN_sec_${SSN}.png"
          done

       else
      
          outfile=`ls sec_Temperature*_contourf_maxdepth_*.png`
          mv $outfile "obs-"${section}"_TEM_sec.png"
          outfile=`ls sec_Salinity*_contourf_maxdepth_*.png`
          mv $outfile "obs-"${section}"_SAL_sec.png"
          outfile=`ls sec_Pot_Density*_contourf_maxdepth_*.png`
          mv $outfile "obs-"${section}"_DEN_sec.png"
       fi
    fi
    
    # MODELS

    if [ ${section} == "osnap-seasonal" ]; then
       for SSN in {"SON","DJF","MAM","JJA"}; do
           # GO8 zps
           inpfile=`ls ${zps_outdir}/nemo_*_1s_average_${SSN}_2014-2018_grid_T.nc`
           python plot_sec.py -zps ${inpfile} -domcfg ${dom_zps} ${section}
           outfile=`ls sec_Temperature*_contourf_maxdepth_*.png`
           mv $outfile "zps"-"${section}_TEM_sec_average_${SSN}_2014-2018.png"
           outfile=`ls sec_Salinity*_contourf_maxdepth_*.png`
           mv $outfile "zps"-"${section}_SAL_sec_average_${SSN}_2014-2018.png"
           outfile=`ls sec_Pot_Density*_contourf_maxdepth_*.png`
           mv $outfile "zps"-"${section}_DEN_sec_average_${SSN}_2014-2018.png"

           # GO8 szt
           inpfile=`ls ${szt_outdir}/nemo_*_1s_average_${SSN}_2014-2018_grid_T.nc`
           python plot_sec.py -szt ${inpfile} -domcfg ${dom_szt} -locmsk ${msk_loc} ${section}
           outfile=`ls sec_Temperature*_contourf_maxdepth_*.png`
           mv $outfile "szt"-"${section}_TEM_sec_average_${SSN}_2014-2018.png"
           outfile=`ls sec_Salinity*_contourf_maxdepth_*.png`
           mv $outfile "szt"-"${section}_SAL_sec_average_${SSN}_2014-2018.png"
           outfile=`ls sec_Pot_Density*_contourf_maxdepth_*.png`
           mv $outfile "szt"-"${section}_DEN_sec_average_${SSN}_2014-2018.png"

           # GO8 MEs
           inpfile=`ls ${MEs_outdir}/nemo_*_1s_average_${SSN}_2014-2018_grid_T.nc`
           python plot_sec.py -mes ${inpfile} -domcfg ${dom_MEs} -locmsk ${msk_loc} ${section}
           outfile=`ls sec_Temperature*_contourf_maxdepth_*.png`
           mv $outfile "MEs"-"${section}_TEM_sec_average_${SSN}_2014-2018.png"
           outfile=`ls sec_Salinity*_contourf_maxdepth_*.png`
           mv $outfile "MEs"-"${section}_SAL_sec_average_${SSN}_2014-2018.png"
           outfile=`ls sec_Pot_Density*_contourf_maxdepth_*.png`
           mv $outfile "MEs"-"${section}_DEN_sec_average_${SSN}_2014-2018.png"
       done

    else
 
       # GO8 zps
       inpfile=`ls ${zps_outdir}/nemo_*_1y_average_2014-2018_grid_T.nc`
       python plot_sec.py -zps ${inpfile} -domcfg ${dom_zps} ${section}
       outfile=`ls sec_Temperature*_contourf_maxdepth_*.png`
       mv $outfile "zps"-"${section}_TEM_sec_average_2014-2018.png"
       outfile=`ls sec_Salinity*_contourf_maxdepth_*.png`
       mv $outfile "zps"-"${section}_SAL_sec_average_2014-2018.png"
       outfile=`ls sec_Pot_Density*_contourf_maxdepth_*.png`
       mv $outfile "zps"-"${section}_DEN_sec_average_2014-2018.png"

       # GO8 szt
       inpfile=`ls ${szt_outdir}/nemo_*_1y_average_2014-2018_grid_T.nc`
       python plot_sec.py -szt ${inpfile} -domcfg ${dom_szt} -locmsk ${msk_loc} ${section}
       outfile=`ls sec_Temperature*_contourf_maxdepth_*.png`
       mv $outfile "szt"-"${section}_TEM_sec_average_2014-2018.png"
       outfile=`ls sec_Salinity*_contourf_maxdepth_*.png`
       mv $outfile "szt"-"${section}_SAL_sec_average_2014-2018.png"
       outfile=`ls sec_Pot_Density*_contourf_maxdepth_*.png`
       mv $outfile "szt"-"${section}_DEN_sec_average_2014-2018.png" 

       # GO8 MEs
       inpfile=`ls ${MEs_outdir}/nemo_*_1y_average_2014-2018_grid_T.nc`
       python plot_sec.py -mes ${inpfile} -domcfg ${dom_MEs} -locmsk ${msk_loc} ${section}
       outfile=`ls sec_Temperature*_contourf_maxdepth_*.png`
       mv $outfile "MEs"-"${section}_TEM_sec_average_2014-2018.png"
       outfile=`ls sec_Salinity*_contourf_maxdepth_*.png`
       mv $outfile "MEs"-"${section}_SAL_sec_average_2014-2018.png"
       outfile=`ls sec_Pot_Density*_contourf_maxdepth_*.png`
       mv $outfile "MEs"-"${section}_DEN_sec_average_2014-2018.png"

    fi
done
