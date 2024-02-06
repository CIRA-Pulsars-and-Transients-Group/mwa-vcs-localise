#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

from mwalib import MetafitsContext
from mwa_hyperbeam import FEEBeam as PrimaryBeam
import numpy as np


def getPrimaryBeamPower(
    metadata: MetafitsContext,
    freq_hz: float,
    alt: float,
    az: float,
    zenithNorm: bool = True,
):
    za = np.pi / 2 - alt
    beam = PrimaryBeam()
    S = np.eye(2) / 2  # equal halves of the "sky" in each X/Y polarisation

    print("... calculating Jones matrices")
    jones = beam.calc_jones_array(
        np.array([az]).flatten(),
        np.array([za]).flatten(),
        freq_hz,
        metadata.delays,
        np.ones_like(metadata.delays),
        zenithNorm,
    )
    print("... creating sky response")
    J = jones.reshape(-1, 2, 2)  # shape = (npix, 2, 2)
    K = np.conjugate(J).T  # shape = (2, 2, npix)
    # This einsum does the following operations:
    # - ij,jkN does the matrix multiplication S @ K, but keeps the dimension N
    # - Nki does the matrix multiplication of each of the N matrices from above, keeping the N outputs
    # - the ->N implies that we want the traces of the N products
    power = np.einsum("Nki,ij,jkN->N", J, S, K).real

    return power
