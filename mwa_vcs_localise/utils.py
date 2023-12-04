#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import numpy as np
from astropy.coordinates import EarthLocation
import astropy.units as u

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
