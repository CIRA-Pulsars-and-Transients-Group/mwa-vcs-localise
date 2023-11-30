#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

from itertools import chain
import numpy as np
from astropy.constants import c as sol


def calcWaveNumbers(freq: float, phi: float, theta: float) -> np.array:
    """Calculate the 3D wavenumbers for a given frequency and sky position.

    :param freq: Radio frequency in Hz.
    :type freq: float
    :param phi: Azimuthal component.
    :type phi: float
    :param theta: Zenith component.
    :type theta: float
    :return: A list wavevectors (X, Y, Z components).
    :rtype: list
    """
    prefactor = 2 * np.pi * freq / sol.value
    kx = prefactor * np.multiply(np.sin(theta), np.cos(phi))
    ky = prefactor * np.multiply(np.sin(theta), np.sin(phi))
    kz = prefactor * np.cos(theta)

    return np.array([kx, ky, kz])


def calcSkyPhase(
    xpos: np.array,
    ypos: np.array,
    zpos: np.array,
    kx: np.array,
    ky: np.array,
    kz: np.array,
    coplanar: bool = True,
) -> np.array:
    """Compute the phase required for each tile for each wavevector (sky position).

    :param xpos: List of tile positions EAST of the array centre.
    :type xpos: np.array
    :param ypos: List of tile positions NORTH of the array centre.
    :type ypos: np.array
    :param zpos: List of tile positions ABOVE the array centre.
    :type zpos: np.array
    :param kx: The X-component of the 3D wavenumbers for a given frequency and set of azimuth/zenith angles.
    :type kx: np.array
    :param ky: The Y-component of the 3D wavenumbers for a given frequency and set of azimuth/zenith angles.
    :type ky: np.array
    :param kz: The Z-component of the 3D wavenumbers for a given frequency and set of azimuth/zenith angles.
    :type kz: np.array
    :param coplanar: Whether we can treat the array as co-planar (i.e., ignore the Z components), defaults to True.
    :type coplanar: bool, optional
    :return: A list of phases for each tile, for each wavevector.
    :rtype: np.array
    """
    if coplanar:
        ph_tile = list(
            chain(
                np.add(np.multiply(kx, x), np.multiply(ky, y))
                for x, y in zip(xpos, ypos)
            )
        )
    else:
        ph_tile = list(
            chain(
                np.add(
                    np.add(np.multiply(kx, x), np.multiply(ky, y)), np.multiply(kz, z)
                )
                for x, y, z in zip(xpos, ypos, zpos)
            )
        )
    return ph_tile
