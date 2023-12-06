#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from astropy.coordinates import EarthLocation, Angle, SkyCoord
import astropy.units as u
from astropy.constants import c as sol
from mwalib import MetafitsContext

MWA_CENTRE_LON = 116.67081524 * u.deg
MWA_CENTRE_LAT = -26.70331940 * u.deg
MWA_CENTRE_H = 377.8269 * u.m
MWA_CENTRE_CABLE_LEN = 0.0 * u.m

MWA_LOCATION = EarthLocation.from_geodetic(
    lon=MWA_CENTRE_LON, lat=MWA_CENTRE_LAT, height=MWA_CENTRE_H
)

SYMMETRY_ANGLES = {
    7: Angle(60 * u.degree),
    8: Angle(51.4 * u.degree),
    9: Angle(45 * u.degree),
    10: Angle(40 * u.degree),
}


def make_grid(az0: float, az1: float, za0: float, za1: float, n: int):
    """Create a mesh-grid in 2D with n cells on each side.

    :param az0: Starting azimuthal coordinate.
    :type az0: float
    :param az1: End azimuthal coordinate.
    :type az1: float
    :param za0: Starting zenith angle coordinate.
    :type za0: float
    :param za1: End zenith angle coordinate.
    :type za1: float
    :param n: Number of cells on each side of the grid.
    :type n: int
    :return: The corresponding azimuth and zenith angle grids.
    :rtype: List[np.array, np.array]
    """
    _az = np.linspace(az0, az1, n)
    _za = np.linspace(za0, za1, n)
    az, za = np.meshgrid(_az, _za)

    return az, za


def form_grid_positions(
    central_coords: SkyCoord,
    max_separation_arcsec: float = 60.0,
    freq_hz: float = 154240000,
    nbeams: int = 7,
    overlap: bool = False,
    verbose: bool = False,
) -> SkyCoord:
    if nbeams not in [7, 8, 9, 10]:
        print(
            "WARNING: Gridding with {0} beams is not currently implemented -"
            " defaulting to 7 (hexagonal packing)".format(nbeams)
        )
        nbeams = 7

    symmetry_angle = Angle(SYMMETRY_ANGLES[nbeams])
    if overlap:
        rho = np.sqrt(3) / 2.0  # optimal coverage for hexagonal circle packing, N=7
    else:
        rho = 1

    fwhm = (max_separation_arcsec * u.arcsecond).to(u.rad)

    nodes = [central_coords]
    for i in range(nbeams - 1):
        nodes.append(nodes[0].directional_offset_by(i * symmetry_angle, rho * fwhm))

    if verbose:
        print("grid #  RA (deg)   Dec (deg)")
        for i, n in enumerate(nodes):
            print(f"{i:<6}  {n.ra.deg:.5f}  {n.dec.deg:.5f}")

    return SkyCoord(nodes)


def find_max_baseline(context: MetafitsContext):
    tile_positions = np.array(
        [
            np.array([rf.east_m, rf.north_m, rf.height_m])
            for rf in context.rf_inputs[::2]
        ]
    )

    hull = ConvexHull(tile_positions)

    # Extract the points forming the hull
    hullpoints = tile_positions[hull.vertices, :]

    # Naive way of finding the best pair in O(H^2) time if H is number of points on
    # hull
    hdist = cdist(hullpoints, hullpoints, metric="euclidean")

    # Get the farthest apart points
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

    return hdist.max(), hullpoints[bestpair[0]], hullpoints[bestpair[1]]
