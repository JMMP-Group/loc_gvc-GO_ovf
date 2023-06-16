#!/bin/bash

cdf_xtrac_brokenline=/local_path_to/CDFTOOLS/bin/cdf_xtrac_brokenline
INPdir=/local_path_to/monthly/model/data/szt
bathy=domain_cfg_szT.nc
osnap=osnap_sec.txt

for TFile in `ls ${INPdir}/*grid-T.nc`; do 
    prefix=${TFile::-10}
    name=`basename ${prefix}`
    UFile=${prefix}_grid-U.nc
    VFile=${prefix}_grid-V.nc
    OFile=output/${name}_osnap.nc

    if [ ! -f "$OFile" ]; then
       ${cdf_xtrac_brokenline} -t  ${TFile} -u ${UFile} -v ${VFile} -b ${bathy} -l ${osnap} -vecrot #-verbose
       mv osnap.nc ${OFile}
       rm osnap_section.dat
       echo "${OFile} ... done!"
    fi
done

