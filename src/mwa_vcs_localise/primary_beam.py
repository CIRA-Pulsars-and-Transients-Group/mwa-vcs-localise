#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import numpy as np
from mwalib import MetafitsContext
from mwa_hyperbeam import FEEBeam as PrimaryBeam


def get_primary_beam_power(
    metadata: MetafitsContext,
    freq_hz: float,
    alt: float | np.ndarray,
    az: float | np.ndarray,
    stokes: str = "I",
    zenith_norm: bool = True,
    show_path: bool = False,
) -> dict[str, np.ndarray]:
    """Calculate the primary beam response (full Stokes) for a given observation
    over a grid of the sky.

    Args:
        metadata (MetafitsContext): A mwalib.MetafitsContext object that contains the
            array configuration and delay settings.
        freq_hz (float): Observing radio frequency, in Hz.
        alt (float | np.ndarray): Desired altitude for the pointing direction, in radians.
        az (float | np.ndarray): Desired azimuth for the pointing direction, in radians.
        stokes (str, optional): Which Stokes parameters to compute and return.
            A string containing some unique combination of "IQUV".
            Values are returned in the order requested here. Defaults to "I".
        zenith_norm (bool, optional): Whether to normalise the primary beam response to
            the value at zenith (maximum sensitivity). Defaults to True.
        show_path (bool, optional): Show the `einsum` optimization path. Defaults to False.

    Raises:
        ValueError: If an invalid Stokes parameter is requested.

    Returns:
        dict[str, np.ndarray]: A dictionary with keys corresponding to the Stokes
            parameters computed, and values being the 2D sky map of the primary beam response.
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
        zenith_norm,
    )
    print("... creating sky response")
    J = jones.reshape(-1, 2, 2)  # shape = (npix, 2, 2)
    K = np.conjugate(J).T  # = J^H, shape = (2, 2, npix)

    # For the coherency matrix products transformed by the Jones matrices, we
    # can use the Pauli spin matrices and simple matrix operations to extract
    # the final Stokes parameters. Effectively using the formalism of the
    # "polarisation measurement equation" of Hamaker (2000) and van Straten (2004).
    rho = dict(
        sI=np.matrix([[1, 0], [0, 1]]),  # sigma0, provides I
        sU=np.matrix([[0, 1], [1, 0]]),  # sigma1, provides U
        sV=np.matrix([[0, -1j], [1j, 0]]),  # sigma2, provides V
        sQ=np.matrix([[1, 0], [0, -1]]),  # sigma3, provides Q
    )
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

    # Here, we figure out the optimal contraction path once, and then just use
    # that for each Stokes parameter. (There is possibly a more efficient combination
    # of operations might scale better, but this is still rapid.)
    einsum_path = np.einsum_path("Nki,ij,jkN->N", J, rho["sI"], K, optimize="optimal")
    if show_path:
        print(einsum_path[0])
        print(einsum_path[1])

    stokes_response = dict()
    for st in stokes:
        # From the Stokes parameter letter, retrieve the correct spin matrix
        rho_mat = rho[f"s{st}"]

        # Determine the scale factor required to apply after matrix operations.
        # Here we use casefold() to ensure comparison is case-agnostic
        if st.casefold() in "IQU".casefold():
            scale = 1 / 2
        elif st.casefold() == "V".casefold():
            scale = -1 / 2
        else:
            print(f"Unrecognized Stokes parameter: {st}!")
            raise ValueError(f"Unrecognized Stokes parameter: st={st}!")

        stokes_response.update(
            {
                f"{st}": scale
                * np.einsum(
                    "Nki,ij,jkN->N", J, rho_mat, K, optimize=einsum_path[0]
                ).real
                # We explicitly take the real part here due to floating-point
                # precision leaving some very small imaginary components in the result
            }
        )

    return stokes_response
