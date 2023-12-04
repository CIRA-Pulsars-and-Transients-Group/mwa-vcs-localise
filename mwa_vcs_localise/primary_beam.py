#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import mwalib
import mwa_hyperbeam
import numpy as np


def getPrimaryBeamPower(
    metadata: mwalib.MetafitsContext, freq_hz: float, alt: float, az: float
):
    za = np.pi / 2 - alt
    beam = mwa_hyperbeam.FEEBeam()
    sky = np.eye(2)
    normZenith = True

    jones = beam.calc_jones_array(
        az,
        za,
        freq_hz,
        metadata.delays,
        np.ones_like(metadata.delays),
        normZenith,
    )
    jmat = np.matrix(jones.reshape((2, 2)))
    power = np.real(np.trace(jmat @ sky @ jmat.H))
    return power
