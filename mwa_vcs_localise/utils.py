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
    nbeams: int = 6,
    nlayers: int = 1,
    overlap: bool = False,
    verbose: bool = False,
) -> SkyCoord:
    """Based on a central sky position, tile beams around it
    with a given separation.

    :param central_coords: The central beam positions, around which to
        tile beams.
    :type central_coords: SkyCoord
    :param max_separation_arcsec: Maximum radial separation (away from
        central point), defaults to 60.0
    :type max_separation_arcsec: float, optional
    :param nbeams: Number of beams to form in the first layer of tiled
        beams, defaults to 6
    :type nbeams: int, optional
    :param nlayers: Number of layers of beams to produce, defaults to 1
    :type nlayers: int, optional
    :param overlap: Whether to allow some overlap of "fwhm" radius
        (effectively shrinks max. separation by ~80%), defaults to False
    :type overlap: bool, optional
    :param verbose: Output extra information, defaults to False
    :type verbose: bool, optional
    :return: A SkyCoord object containing all produced tiling beams.
    :rtype: SkyCoord
    """
    symmetry_angle = Angle(360 * u.deg / (nbeams - 1))
    if overlap:
        rho = np.sqrt(3) / 2.0  # optimal coverage for hexagonal circle packing, N=7
    else:
        rho = 1

    fwhm = (max_separation_arcsec * u.arcsecond).to(u.rad)

    nodes = [central_coords]
    for j in range(nlayers):
        nnodes_this_layer = (j + 1) * (nbeams - 1)
        if verbose:
            print(f"number of nodes to make on layer {j}: {nnodes_this_layer}")
        for i in range(nnodes_this_layer):
            pa = i * symmetry_angle / (j + 1)
            r = rho * fwhm * (j + 1)
            node = nodes[0].directional_offset_by(pa, r)
            if verbose:
                print(f"made node: r={r} pa={pa}")
            nodes.append(node)

    if verbose:
        print("grid #  RA (deg)   Dec (deg)")
        for i, n in enumerate(nodes):
            print(f"{i:<6}  {n.ra.deg:.5f}  {n.dec.deg:.5f}")

    return SkyCoord(nodes)


def find_max_baseline(context: MetafitsContext) -> list:
    """Use a Convex Hull method to calculate the maximum distance
    between two tiles given their 3D coordinates.

    :param context: A mwalib.MetafitsContext object that contains
        tile-position information.
    :type context: MetafitsContext
    :return: The maximum distance, and corresponding pair of
        coordinates.
    :rtype: list, 3-elements
    """
    tile_positions = np.array(
        [
            np.array([rf.east_m, rf.north_m, rf.height_m])
            for rf in context.rf_inputs[::2]
        ]
    )
    # Create the convex hull
    hull = ConvexHull(tile_positions)

    # Extract the points forming the hull
    hullpoints = tile_positions[hull.vertices, :]

    # Naive way of finding the best pair in O(H^2) time if H is number
    # of points on the hull
    hdist = cdist(hullpoints, hullpoints, metric="euclidean")

    # Get the farthest apart points
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

    return [hdist.max(), hullpoints[bestpair[0]], hullpoints[bestpair[1]]]
