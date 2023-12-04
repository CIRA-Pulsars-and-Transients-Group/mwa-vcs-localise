#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import mwalib
import mwa_hyperbeam
import numpy as np
from .utils import make_grid


def getPrimaryBeamPower(
    metadata: mwalib.MetafitsContext,
    freq_hz: float,
    alt: float,
    az: float,
    zenithNorm: bool = True,
):
    za = np.pi / 2 - alt
    beam = mwa_hyperbeam.FEEBeam()
    sky = np.eye(2)

    jones = beam.calc_jones_array(
        np.array([az]),
        np.array([za]),
        freq_hz,
        metadata.delays,
        np.ones_like(metadata.delays),
        zenithNorm,
    )
    jmat = np.matrix(jones.reshape((2, 2)))
    power = np.real(np.trace(jmat @ sky @ jmat.H))
    return power


def getPrimaryBeamPower_2D(
    metadata: mwalib.MetafitsContext,
    freq_hz: float,
    alt0: float,
    alt1: float,
    az0: float,
    az1: float,
    ncells: int,
    zenithNorm: bool = True,
):
    za0 = np.pi / 2 - alt0
    za1 = np.pi / 2 - alt1
    az, za = make_grid(az0, az1, za0, za1, ncells)

    beam = mwa_hyperbeam.FEEBeam()
    sky = np.eye(2)

    jones = beam.calc_jones_array(
        az.flatten(),
        za.flatten(),
        freq_hz,
        metadata.delays,
        np.ones_like(metadata.delays),
        zenithNorm,
    )
    jmat = np.matrix(jones.reshape((2, 2)))
    power = np.real(np.trace(jmat @ sky @ jmat.H))

    return az, za, power.reshape(az.shape)
