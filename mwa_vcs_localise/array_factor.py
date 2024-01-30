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
    # At this stage, the shape of target_w = (nant, n_ra, n_dec) and while the shape of look_w = (nant,)
    sum_over_antennas = np.tensordot(np.conjugate(look_w), target_w, axes=1)
    # From the numpy.tensordot documentation:
    #    The third argument can be a single non-negative integer_like scalar, N;
    #    if it is such, then the last N dimensions of a and the first N dimensions
    #    of b are summed over.

    # The array factor power is normalised to the number of elements
    # included in the sum (i.e., length of the `look_w` vector).
    afp = (np.absolute(sum_over_antennas) / look_w.size) ** 2
    return afp
