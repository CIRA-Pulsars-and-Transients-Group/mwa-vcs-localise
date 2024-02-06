#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import numpy as np
from mwalib import MetafitsContext
from astropy.constants import c as sol

from .utils import MWA_CENTRE_CABLE_LEN


def extractWorkingTilePositions(metadata: MetafitsContext):
    """Extract tile position information required for beamforming and/or
    computing the array factor quantity from a metafits structure.
    Flagged tiles are automatically excluded from the result.

    :param metadata: An MWALIB MetafitsContext structure
                     containing the array layout information.
    :type metadata: MetafitsContext
    :return: Working tile positions and electrical lengths for
             beamforming. Formatted as an array of arrays, where
             each item in the outer array is:
                [east_m, north_m, height_m, electrical_length_m]
             for a single tile.
    :rtype: np.ndarray
    """
    # We only care about the tiles, not polarisations, so skip the Y pol
    rf_inputs = metadata.rf_inputs[::2]

    # Gather the tile positions into a "vector" for each tile
    tile_positions = np.array(
        [
            np.array(
                [
                    rf.east_m,
                    rf.north_m,
                    rf.height_m,
                    rf.electrical_length_m - MWA_CENTRE_CABLE_LEN.value,
                ]
            )
            for rf in rf_inputs
        ]
    )

    # Gather the flagged tile information from the metafits information
    # and remove those tiles from the above vector
    tile_flags = np.array([rf.flagged for rf in rf_inputs])
    tile_positions = np.delete(tile_positions, np.where(tile_flags == True), axis=0)

    return tile_positions


def calcGeometricDelays(positions: np.ndarray, freq_hz: float, alt: float, az: float):
    """Compute the geometric delay phases for each element position in order to
    "phase up" to the provided position at a specific frequency. These are the
    phasors used in a beamforming operation.

    :param positions: An array or element position vectors, including their
                      equivalent electrical length, in metres.
    :type positions: np.ndarray
    :param freq_hz: Observing radio frequency, in Hz.
    :type freq_hz: float
    :param alt: Desired altitude for the pointing direction, in radians. Can be an array.
    :type alt: np.ndarray, float
    :param az: Desired azimuth for the pointing direction, in radians. Can be an array.
    :type az: np.ndarray, float
    :return: The required phasors needed to rotate the element patterns to each requested az/alt pair.
    :rtype: np.ndarray, complex
    """
    # Create the unit vector(s)
    u = np.array(
        [
            np.cos(alt) * np.sin(az),  # unit E
            np.cos(alt) * np.cos(az),  # unit N
            np.sin(alt),  # unit H
            -np.ones_like(alt),  # cable length (-ve as it is subtracted)
        ]
    )

    # Compute the equivalent delay length for each tile
    # (Use tensor dot product so we can choose to keep the
    # dimensionality of the alt/az grid and continue using
    # broadcasting rules efficiently.)
    w = np.tensordot(positions, u, axes=1)

    # Convert to a time delay
    dt = w / sol.value

    # Construct the phasor
    phase = 2 * np.pi * freq_hz * dt
    phasor = np.exp(1.0j * phase)

    return phasor


def calcArrayFactorPower(look_w, target_w):
    """Compute the array factor power from a given pointing phasor
    and one or more target directions.

    :param look_w: The complex phasor representing the tile phases
        in the desired "look direction".
    :type look_w: np.array, complex
    :param target_w: The complex phasor(s) representing the tile
        phases required to look in the desired sample directions.
    :type target_w: np.ndarray, complex
    :return: The absolute array factor power, for each given
        target direction.
    :rtype: np.ndarray
    """
    # At this stage, the shape of target_w = (nant, n_ra, n_dec) and while the shape of look_w = (nant,)
    print("... summing over antennas")
    sum_over_antennas = np.tensordot(np.conjugate(look_w), target_w, axes=1)
    # From the numpy.tensordot documentation:
    #    The third argument can be a single non-negative integer_like scalar, N;
    #    if it is such, then the last N dimensions of a and the first N dimensions
    #    of b are summed over.

    # The array factor power is normalised to the number of elements
    # included in the sum (i.e., length of the `look_w` vector).
    print("... converting to power")
    afp = (np.absolute(sum_over_antennas) / look_w.size) ** 2
    return afp
