#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

from mwalib import MetafitsContext
from mwa_hyperbeam import FEEBeam as PrimaryBeam
import numpy as np
from .utils import make_grid


def getPrimaryBeamPower(
    metadata: MetafitsContext,
    freq_hz: float,
    alt: float,
    az: float,
    zenithNorm: bool = True,
):
    za = np.pi / 2 - alt
    beam = PrimaryBeam()
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
    metadata: MetafitsContext,
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

    beam = PrimaryBeam()
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
