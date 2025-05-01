#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import numpy as np
from mwalib import MetafitsContext, Pol
from astropy.constants import c as sol

from .utils import MWA_CENTRE_CABLE_LEN


def extract_working_tile_positions(
    metadata: MetafitsContext,
) -> tuple[np.ndarray, int, int]:
    """Extract tile position information required for beamforming and/or
    computing the array factor quantity from a metafits structure.

    Flagged tiles are automatically excluded from the result.


    Args:
        metadata (MetafitsContext): An MWALIB MetafitsContext structure
            containing the array layout information.

    Returns:
        tuple[np.ndarray, int, int]: A tuple containing:
            (1) Working tile positions and electrical lengths for
                beamforming, formatted as an array of arrays, where
                each item in the outer array is:
                    [east_m, north_m, height_m, electrical_length_m]
                for a single tile.
             (2) The number of unflagged tiles, and
             (3) The number of flagged tiles.
    """

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
            for rf in metadata.rf_inputs
            if rf.pol == Pol.X
        ]
    )

    # Gather the flagged tile information from the metafits information
    # and remove those tiles from the above vector
    tile_flags = np.array([rf.flagged for rf in metadata.rf_inputs if rf.pol == Pol.X])
    tile_positions = np.delete(tile_positions, np.where(tile_flags & True), axis=0)

    num_ok_tiles = (~tile_flags).sum()
    num_bad_tiles = (tile_flags).sum()

    return tile_positions, num_ok_tiles, num_bad_tiles


def calc_geometric_delays(
    positions: np.ndarray,
    freq_hz: float,
    alt: float | np.ndarray,
    az: float | np.ndarray,
) -> np.ndarray:
    """Compute the geometric delay phases for each element position in order to
    "phase up" to the provided position at a specific frequency. These are the
    phasors used in a beamforming operation.

    Args:
        positions (np.ndarray): An array or element position vectors, including their
            equivalent electrical length, in metres.
        freq_hz (float): Observing radio frequency, in Hz.
        alt (float | np.ndarray): Desired altitude for the pointing direction, in radians.
        az (float | np.ndarray): Desired azimuth for the pointing direction, in radians.

    Returns:
        np.ndarray: The required phasors needed to rotate the element patterns to
            each requested az/alt pair.
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
    # From the numpy.tensordot documentation:
    #    The third argument can be a single non-negative integer_like scalar, N;
    #    if it is such, then the last N dimensions of a and the first N dimensions
    #    of b are summed over.

    # Convert to a time delay
    dt = w / sol.value

    # Construct the phasor
    phase = 2 * np.pi * freq_hz * dt
    phasor = np.exp(1.0j * phase)

    return phasor


def calc_array_factor_power(look_w: np.ndarray, target_w: np.ndarray) -> np.ndarray:
    """Compute the array factor power from a given pointing phasor
    and one or more target directions.

    Args:
        look_w (np.ndarray): The complex phasor representing the tile phases
            in the desired "look direction".
        target_w (np.ndarray): The complex phasor(s) representing the tile
            phases required to look in the desired sample directions.

    Returns:
        np.ndarray: The absolute array factor power, for each given
            target direction.
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
