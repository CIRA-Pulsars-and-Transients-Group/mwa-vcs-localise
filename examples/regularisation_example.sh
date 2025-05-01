#!/bin/bash 

obsid=1226062160
metafits="${obsid}.metafits"
obstime="2018-11-12T13:30:00"
singlefreq=154240000
detection_file="initial_detections_0026.txt"


# Regenerate almost identical subfigures as in Appendix Figure 1.
for reg in none gaussian tab;do
    mwa_tab_loc \
        -m ${metafits} \
        -f ${singlefreq} \
        -t ${obstime} \
        --detfile ${detection_file} \
        --gridbox '00:08:00 -21:30:00 00:37:00 -18:30:00 10 10' \
        --plot \
        --localise \
        --regularise ${reg} \
        --truth '00:26:36.3 -19:55:59.3' \
        --loc_fig_lims "5 7 -20.5 -18.5"
    
    mv localisation.png init_localisation_0026.${reg}.png
done

