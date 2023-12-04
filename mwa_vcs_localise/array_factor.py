#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import numpy as np
import mwalib
from astropy.constants import c as sol

from .utils import MWA_CENTRE_LON, MWA_CENTRE_LAT, MWA_CENTRE_H, MWA_CENTRE_CABLE_LEN


def calcGeometricDelays(
    metadata: mwalib.MetafitsContext, freq_hz: float, alt: float, az: float
):
    unit_N = np.cos(alt) * np.cos(az)
    unit_E = np.cos(alt) * np.sin(az)
    unit_H = np.sin(alt)
    rf_inputs = metadata.rf_inputs[::2]
    phi = []
    for rf in rf_inputs:
        e = rf.east_m - MWA_CENTRE_LON.value
        n = rf.north_m - MWA_CENTRE_LAT.value
        h = rf.height_m - MWA_CENTRE_H.value
        ell = rf.electrical_length_m - MWA_CENTRE_CABLE_LEN.value

        w = e * unit_E + n * unit_N + h * unit_H
        dt = (w - ell) / sol.value
        phase = 2 * np.pi * freq_hz * dt
        phi.append(np.cos(phase) + 1.0j * np.sin(phase))

    return np.array(phi)


def calcArrayFactorPower(look_w, target_w):
    sum_over_antennas = np.sum(np.conjugate(look_w) * target_w)
    afp = (np.absolute(sum_over_antennas) / look_w.size) ** 2
    return afp
