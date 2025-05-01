#!/usr/bin/env bash

obsid=1253471952
metafits="${obsid}.metafits"
obstime="2019-09-25T18:58:54"
lookdir="03:00:000_-55:00:00.00"
singlefreq=154000000


# Get the metafits files from the MWA web service.
wget "http://ws.mwatelescope.org/metadata/fits?obs_id=${obsid}" -O ${metafits}


# Paper Figure 1 (array layout and baseline distribution) & Figure 2 (zoom of central and first sidelobes of tied-array beam) WITHOUT primary beam effects
mwa_tab_loc \
    -m "${metafits}" \
    -f ${singlefreq} \
    -t "${obstime}" \
    -L "${lookdir}" \
    --gridbox "02:50:00 -56:40:00 03:10:00 -53:20:00 10 10" \
    --nopb \
    --plot

mv ${obsid}_tiedarray_beam_nopb.png ${obsid}_tiedarray_beam_nopb_zoom.png

# Lower panel of Paper Figure 2 (tied-array beam pattern WITH primary beam effects)
mwa_tab_loc \
    -m "${metafits}" \
    -f ${singlefreq} \
    -t "${obstime}" \
    -L "${lookdir}" \
    --gridbox "02:50:00 -56:40:00 03:10:00 -53:20:00 10 10" \
    --plot

mv ${obsid}_tiedarray_beam_pb.png ${obsid}_tiedarray_beam_pb_zoom.png
mv ${obsid}_pb.png ${obsid}_pb_zoom.png

