#!/usr/bin/env bash

obsid=1258221008
metafits="${obsid}.metafits"
obstime="2019-11-19T18:10:00"
singlefreq=154240000
detection_file="detections_0452.txt"

# Get the metafits files from the MWA web service.
if [ ! -f ${metafits} ];then
    wget "http://ws.mwatelescope.org/metadata/fits?obs_id=${obsid}" -O ${metafits}
fi

# Generate the localisation plot.
mwa_tab_loc \
    -m ${metafits} \
    -f ${singlefreq} \
    -t ${obstime} \
    --detfile ${detection_file} \
    --use-wcs --wcs-pixel-size 5 \
    --zoom \
    --plot \
    --localise \
    --truth '04:52:00.7 -34:18:42.0'

mv localisation.png localisation_psr0452.png 
mv covariance.png covariance_psr0452.png
