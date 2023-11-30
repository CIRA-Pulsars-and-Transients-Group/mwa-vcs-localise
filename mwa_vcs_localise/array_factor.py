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
    tile_x: np.array,
    tile_y: np.array,
    tile_z: np.array,
    sky_kx: np.array,
    sky_ky: np.array,
    sky_kz: np.array,
    coplanar: bool = True,
) -> np.array:
    """Compute the phase required for each tile for each wavevector (sky position).

    :param tile_x: List of tile positions EAST of the array centre.
    :type tile_x: np.array
    :param tile_y: List of tile positions NORTH of the array centre.
    :type tile_y: np.array
    :param tile_z: List of tile positions ABOVE the array centre.
    :type tile_z: np.array
    :param sky_kx: The X-components of the 3D wavenumbers for a given
                    frequency and set of sky positions.
    :type sky_kx: np.array
    :param sky_ky: The Y-components of the 3D wavenumbers for a given
                    frequency and set of sky positions.
    :type sky_ky: np.array
    :param sky_kz: The Z-components of the 3D wavenumbers for a given
                    frequency and set of sky positions.
    :type sky_kz: np.array
    :param coplanar: Whether we can treat the array as co-planar
                    (i.e., ignore the Z components), defaults to True.
    :type coplanar: bool, optional
    :return: A list of phases for each tile, for each wavevector.
    :rtype: np.array
    """
    if coplanar:
        # The "tile phase" represents the response at a given sky position at a certain frequency
        ph_tile = list(
            chain(
                np.add(np.multiply(sky_kx, x), np.multiply(sky_ky, y))
                for x, y in zip(tile_x, tile_y)
            )
        )
    else:
        ph_tile = list(
            chain(
                np.add(
                    np.add(np.multiply(sky_kx, x), np.multiply(sky_ky, y)),
                    np.multiply(sky_kz, z),
                )
                for x, y, z in zip(tile_x, tile_y, tile_z)
            )
        )
    return ph_tile


def calcArrayFactor(ph_tiles: np.array, ph_targets: np.array) -> np.array:
    """Compute the array factor (complex field) for each tile towards each target sky position.

    :param ph_tiles: List of the sky phases for the tiles
    :type ph_tiles: np.array
    :param ph_targets: List of the target phases on the sky
    :type ph_targets: np.array
    :return: The complex-valued array factor for each sky position.
    :rtype: np.array
    """
    # Compute the array factor for each tile
    array_factor_tiles = list(
        chain(
            np.multiply(
                np.cos(ph_tile) + 1.0j * np.sin(ph_tile),
                np.cos(ph_target) - 1.0j * np.sin(ph_target),
            )
            for ph_tile, ph_target in zip(ph_tiles, ph_targets)
        )
    )
    # Sum over the tiles
    array_factor = np.sum(array_factor_tiles, axis=0)

    # Normalise such that the power is unity at the pointing position
    array_factor = np.divide(array_factor, len(ph_tiles))

    return array_factor


def calcArrayFactorPower(ph_tiles: np.array, ph_targets: np.array) -> np.array:
    """Compute the array factor power for each tile towards each target sky position.

    :param ph_tiles: List of the sky phases for the tiles
    :type ph_tiles: np.array
    :param ph_targets: List of the target phases on the sky
    :type ph_targets: np.array
    :return: The real-valued array factor power for each sky position.
    :rtype: np.array
    """
    array_factor = calcArrayFactor(ph_tiles, ph_targets)
    array_factor_power = np.abs(array_factor) ** 2

    return array_factor_power
