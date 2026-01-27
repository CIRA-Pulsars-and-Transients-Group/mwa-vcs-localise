#!/bin/bash 

obsid=1226062160
metafits="${obsid}.metafits"
obstime="2018-11-12T13:30:00"
singlefreq=154240000
detection_file="initial_detections_0026.txt"

# Get the metafits files from the MWA web service.
if [ ! -f $metafits ];then
    wget "http://ws.mwatelescope.org/metadata/fits?obs_id=${obsid}" -O ${metafits}
fi

# Regenerate subfigures as in Appendix Figure 1.
for reg in none gaussian tab;do
    mwa_tab_loc \
        -m ${metafits} \
        -f ${singlefreq} \
        -t ${obstime} \
        --detfile ${detection_file} \
        --use_wcs --wcs_pixel_size 5 --wcs_grid_size 1536 1536 \
        --plot \
        --localise \
        --regularise ${reg} \
        --truth '00:26:36.3 -19:55:59.3'
    
    mv localisation.png init_localisation_0026.${reg}.png
done

