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
    stokes: str = "I",
    zenithNorm: bool = True,
    show_path: bool = False,
):
    """Calculate the primary beam response (full Stokes) for a
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
    :param stokes: Which Stokes parameters to compute and return.
                   A string containing some unique combination of "IQUV".
                   Values are returned in the order requested here.
    :type stokes: str
    :param zenithNorm: Whether to normalise the primary beam response to
                       the value at zenith (maximum sensitivity).
                       Defaults to True.
    :type zenithNorm: bool, optional
    :param show_path: Show the einsum optimization path. Defaults to False.
    :type show_path: bool, optional
    :return: The primary beam response over the provided sky positions.
             The position axis is flattened, thus needs to be reshaped
             based on the input az/alt arguments.
    :rtype: np.ndarray
    """
    za = np.pi / 2 - alt
    beam = PrimaryBeam()

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
    K = np.conjugate(J).T  # = J^H, shape = (2, 2, npix)

    # For the coherency matrix products transformed by the Jones matrices, we
    # can use the Pauli spin matrices and simple matrix operations to extract
    # the final Stokes parameters. Effectively using the formalism of the
    # "polarisation measurement equation" of Hamaker (2000) and van Straten (2004).
    rho = dict(
        S0=np.matrix([[1, 0], [0, 1]]),  # sigma0 (identity)
        S1=np.matrix([[0, 1], [1, 0]]),  # sigma1
        S2=np.matrix([[0, -1j], [1j, 0]]),  # sigma2
        S3=np.matrix([[1, 0], [0, -1]]),  # sigma3
    )
    # map the Stokes parameters to their spin matrix transforms
    stokes_to_rho = dict(I="S0", Q="S3", U="S1", V="S2")

    # Multiplying the above spin matrices on the left by the Jones matrix,
    # and on the right by the Hermitian transpose of the Jones matrix will
    # retrieve the Stokes response of the instrument (modulo a scaling factor).
    # i.e., for each of the N sky positions sampled,
    #
    #   Tr[ J @ S0 @ K ] = 2I
    #   Tr[ J @ S1 @ K ] = 2U
    #   Tr[ J @ S2 @ K ] = -i(U - iV) + i(U + iV) = -2V
    #   Tr[ J @ S3 @ K ] = 2Q
    #
    # where Tr is the trace operator, @ implies matrix multiplication,
    # and "i" is the imaginary unit.

    # Here, we figure out the optimal contraction path once, and then just use that for each Stokes parameter
    # (Possibly a more efficient combination of operations might scale better, but this is still rapid.)
    einsum_path = np.einsum_path("Nki,ij,jkN->N", J, rho["S0"], K, optimize="optimal")
    if show_path:
        print(einsum_path[0])
        print(einsum_path[1])
    # This einsum does the following operations:
    # - N is our "batch" dimension, so we can do a batch of N matrix multiplications
    # - first, we do the multiplication of N (k x i) Jones matrices onto our (i x j) Pauli matrix
    # - then we do the multiplication of the N (j x k) composite matrices onto the inverse Jones matrix (j x k)
    # - finally, the "->N" symbol implies the trace (sum of diagonals) of each N matrices

    stokes_response = []
    for st in stokes:
        rho_mat = rho[stokes_to_rho[st]]
        # Use str.casefold() to ensure comparison is case-agnostic
        if st.casefold() in "IQU".casefold():
            scale = 1 / 2
        elif st.casefold() == "V".casefold():
            scale = -1 / 2
        else:
            print(f"Unrecognized Stokes parameter: {st}!")
            raise ValueError(f"Unrecognized Stokes parameter: st={st}!")

        stokes_response.append(
            scale
            * np.einsum(
                "Nki,ij,jkN->N",
                J,
                rho_mat,
                K,
                optimize=einsum_path[0],
            ).real
            # We explicitly take the real part here due to floating-point
            # precision leaving some very small imaginary components in the result
        )

    return stokes_response
