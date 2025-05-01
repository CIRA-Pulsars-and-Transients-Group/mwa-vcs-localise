#!/usr/bin/env bash

obsid=1226062160
metafits="${obsid}.metafits"
obstime="2018-11-12T13:30:00"
singlefreq=154240000
detection_file="detections_0026.txt"


# Get the metafits files from the MWA web service.
wget "http://ws.mwatelescope.org/metadata/fits?obs_id=${obsid}" -O ${metafits}

# Generate the localisation plot.
mwa_tab_loc \
    -m ${metafits} \
    -f ${singlefreq} \
    -t ${obstime} \
    --detfile ${detection_file} \
    --gridbox '00:08:00 -21:30:00 00:37:00 -18:30:00 10 10' \
    --plot \
    --localise \
    --truth '00:26:36.3 -19:55:59.3'

mv localisation.png localisation_psr0026.png 
mv covariance.png covariance_psr0026.png

