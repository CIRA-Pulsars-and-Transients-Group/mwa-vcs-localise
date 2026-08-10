########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import numpy as np
from astropy.constants import c as sol
from mwalib import MetafitsContext, Pol

from .utils import MWA_CENTRE_CABLE_LEN


def extract_working_tile_positions(
    metadata: MetafitsContext,
    extra_tile_flags: list[str] | None = None,
    exclude_flagged: bool = True,
) -> tuple[np.ndarray, int, int]:
    """Extract tile positions used for beamforming from metafits metadata.

    :param metadata: MWALIB metadata containing array layout and tile state.
    :type metadata: MetafitsContext
    :param extra_tile_flags: Additional tile names or IDs to flag.
    :type extra_tile_flags: list[str] | None
    :param exclude_flagged: If True, omit flagged tiles from returned positions.
    :type exclude_flagged: bool
    :returns: Tile position vectors ``[east, north, height, cable_length]``, the
        number of unflagged tiles, and the number of flagged tiles.
    :rtype: tuple[np.ndarray, int, int]
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
    tile_flags = np.array(
        [rf.flagged for rf in metadata.rf_inputs if rf.pol == Pol.X]
    )
    if extra_tile_flags is not None:
        itile = 0
        for rf in metadata.rf_inputs:
            if rf.pol != Pol.X:
                continue
            if (
                rf.tile_name in extra_tile_flags
                or str(rf.tile_id) in extra_tile_flags
            ):
                tile_flags[itile] = True
            itile += 1

    if exclude_flagged:
        tile_positions = np.delete(
            tile_positions, np.where(tile_flags & True), axis=0
        )
    else:
        print("Not removing flagged tiles from list.")

    num_ok_tiles = (~tile_flags).sum()
    num_bad_tiles = (tile_flags).sum()

    return tile_positions, num_ok_tiles, num_bad_tiles


def calc_geometric_delays(
    positions: np.ndarray,
    freq_hz: float,
    alt: float | np.ndarray,
    az: float | np.ndarray,
) -> np.ndarray:
    """Compute geometric delay phasors for one or more sky directions.

    :param positions: Tile position vectors including electrical length in
        metres.
    :type positions: np.ndarray
    :param freq_hz: Observing frequency in Hz.
    :type freq_hz: float
    :param alt: Altitude in radians.
    :type alt: float | np.ndarray
    :param az: Azimuth in radians.
    :type az: float | np.ndarray
    :returns: Complex phasors for each requested direction.
    :rtype: np.ndarray
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


def calc_array_factor_power(
    look_w: np.ndarray, target_w: np.ndarray
) -> np.ndarray:
    """Compute array-factor power for sampled sky directions.

    :param look_w: Complex phasor vector for the look direction.
    :type look_w: np.ndarray
    :param target_w: Complex phasors for sampled target directions.
    :type target_w: np.ndarray
    :returns: Normalised array-factor power for each target direction.
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
