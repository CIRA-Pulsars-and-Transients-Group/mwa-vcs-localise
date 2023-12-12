#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import numpy as np
from mwalib import MetafitsContext
from astropy.constants import c as sol

from .utils import MWA_CENTRE_CABLE_LEN


def calcGeometricDelays(
    metadata: MetafitsContext, freq_hz: float, alt: float, az: float
):
    unit_N = np.cos(alt) * np.cos(az)
    unit_E = np.cos(alt) * np.sin(az)
    unit_H = np.sin(alt)
    rf_inputs = metadata.rf_inputs[::2]
    phi = []
    for rf in rf_inputs:
        e = rf.east_m
        n = rf.north_m
        h = rf.height_m
        ell = rf.electrical_length_m - MWA_CENTRE_CABLE_LEN.value

        w = e * unit_E + n * unit_N + h * unit_H
        dt = (w - ell) / sol.value
        phase = 2 * np.pi * freq_hz * dt
        phi.append(np.cos(phase) + 1.0j * np.sin(phase))

    return np.array(phi)


def calcArrayFactorPower(look_w, target_w):
    """Compute the array factor power from a given pointing phasor
    and one or more target directions.

    :param look_w: The complex phasor representing the tile phases
        in the desired "look direction".
    :type look_w: np.array, complex
    :param target_w: The complex phasor(s) representing the tile
        phases required to look in the desired sample directions.
    :type target_w: np.ndarray, complex
    :return: The absolute array factor power, for each given
        target direction.
    :rtype: np.ndarray
    """
    sum_over_antennas = np.dot(np.conjugate(look_w), target_w)
    # From the numpy.dot documentation:
    #
    #    "If `a` is an N-D array and `b` is a 1-D array,
    #     it is a sum product over the last axis of `a` and `b`."
    #
    # which is ideal in our case because `a` is our look direction
    # and `b` is our (potentially many) target directions.

    # The array factor power is normalised to the number of elements
    # included in the sum (i.e., length of the `look_w` vector).
    afp = (np.absolute(sum_over_antennas) / look_w.size) ** 2
    return afp
