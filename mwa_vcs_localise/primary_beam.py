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
        np.array([az]).flatten(),
        np.array([za]).flatten(),
        freq_hz,
        metadata.delays,
        np.ones_like(metadata.delays),
        zenithNorm,
    )
    jmats = [np.matrix(j.reshape((2, 2))) for j in jones]
    power = [np.real(np.trace(jmat @ sky @ jmat.H)) for jmat in jmats]
    return np.array(power)
