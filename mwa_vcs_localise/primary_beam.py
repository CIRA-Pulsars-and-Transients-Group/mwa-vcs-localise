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
    """Calculate the primary beam response (Stokes I only) for a
    given observation over a grid of the sky.

    :param metadata: A mwalib.MetafitsContext object that contains the
                     array configuration and delay settings.
    :type metadata: MetafitsContext
    :param freq_hz: Observing radio frequency, in Hz.
    :type freq_hz: float
    :param alt: Desired altitude for the pointing direction, in radians.
                Can be an array.
    :type alt: np.ndarray, float
    :param az: Desired azimuth for the pointing direction, in radians.
               Can be an array.
    :type az: np.ndarray, float
    :param zenithNorm: Whether to normalise the primary beam response to
                       the value at zenith (maximum sensitivity).
                       Defaults to True.
    :type zenithNorm: bool, optional
    :return: The primary beam response over the provided sky positions.
             The position axis is flattened, thus needs to be reshaped
             based on the input az/alt arguments.
    :rtype: np.ndarray
    """
    za = np.pi / 2 - alt
    beam = PrimaryBeam()
    S = np.eye(2) / 2  # equal halves of the "sky" in each X/Y polarisation
    # Different matrices would need to be used to construct the other 3 Stokes
    # parameters, but the operations themselves would be identical to below.

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
    power = np.einsum("Nki,ij,jkN->N", J, S, K, optimize=True).real

    return power
