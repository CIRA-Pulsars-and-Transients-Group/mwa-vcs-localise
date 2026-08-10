########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import numpy as np
from mwa_hyperbeam import FEEBeam as PrimaryBeam
from mwalib import MetafitsContext


def get_primary_beam_power(
    metadata: MetafitsContext,
    freq_hz: float,
    alt: float | np.ndarray,
    az: float | np.ndarray,
    stokes: str = "I",
    zenith_norm: bool = True,
    show_path: bool = False,
) -> dict[str, np.ndarray]:
    """Calculate primary-beam Stokes response across sampled sky positions.

    :param metadata: MWALIB metadata containing delay settings.
    :type metadata: MetafitsContext
    :param freq_hz: Observing frequency in Hz.
    :type freq_hz: float
    :param alt: Altitude in radians.
    :type alt: float | np.ndarray
    :param az: Azimuth in radians.
    :type az: float | np.ndarray
    :param stokes: Requested Stokes parameters as a unique subset of ``IQUV``.
    :type stokes: str
    :param zenith_norm: If True, normalise by zenith response.
    :type zenith_norm: bool
    :param show_path: If True, print ``einsum`` optimisation details.
    :type show_path: bool
    :raises ValueError: If a requested Stokes parameter is not recognised.
    :returns: Mapping from each requested Stokes key to a flattened response map.
    :rtype: dict[str, np.ndarray]
    """

    za = np.pi / 2 - alt
    beam = PrimaryBeam(None)

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
    rho = {
        "sI": np.matrix([[1, 0], [0, 1]]),  # sigma0, provides I
        "sU": np.matrix([[0, 1], [1, 0]]),  # sigma1, provides U
        "sV": np.matrix([[0, -1j], [1j, 0]]),  # sigma2, provides V
        "sQ": np.matrix([[1, 0], [0, -1]]),  # sigma3, provides Q
    }
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
    einsum_path = np.einsum_path(
        "Nki,ij,jkN->N", J, rho["sI"], K, optimize="optimal"
    )
    if show_path:
        print(einsum_path[0])
        print(einsum_path[1])

    stokes_response = {}
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
            raise ValueError(f"Unrecognised Stokes parameter: st={st}!")

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
