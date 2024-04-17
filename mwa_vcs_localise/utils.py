#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
import multiprocessing

import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from astropy.coordinates import EarthLocation
import astropy.units as u
from mwalib import MetafitsContext

MWA_CENTRE_LON = 116.67081524 * u.deg
MWA_CENTRE_LAT = -26.70331940 * u.deg
MWA_CENTRE_H = 377.8269 * u.m
MWA_CENTRE_CABLE_LEN = 0.0 * u.m

MWA_LOCATION = EarthLocation.from_geodetic(
    lon=MWA_CENTRE_LON, lat=MWA_CENTRE_LAT, height=MWA_CENTRE_H
)


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
            for rf in context.rf_inputs[: context.num_ants]
        ]
    )
    tile_flags = np.array([rf.flagged for rf in context.rf_inputs[: context.num_ants]])
    tile_positions = np.delete(tile_positions, np.where(tile_flags == True), axis=0)

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


def plot_array_layout(context: MetafitsContext) -> None:
    tile_positions = np.array(
        [
            np.array([rf.east_m, rf.north_m, rf.height_m])
            for rf in context.rf_inputs[: context.num_ants]
        ]
    )
    tile_flags = np.array([rf.flagged for rf in context.rf_inputs[: context.num_ants]])
    max_baseline = find_max_baseline(context)[0]

    okay_tiles_n = np.ma.masked_array(tile_positions[:, 1], mask=tile_flags)
    okay_tiles_e = np.ma.masked_array(tile_positions[:, 0], mask=tile_flags)
    bad_tiles_n = np.ma.masked_array(tile_positions[:, 1], mask=~tile_flags)
    bad_tiles_e = np.ma.masked_array(tile_positions[:, 0], mask=~tile_flags)

    fig = plt.figure()
    fig.add_subplot()
    plt.scatter(okay_tiles_e, okay_tiles_n, zorder=1000, s=10, marker="x", color="k")
    plt.scatter(bad_tiles_e, bad_tiles_n, zorder=1000, s=10, marker="x", color="r")
    plt.xlim(-410, 410)
    plt.ylim(50, 600)
    plt.xlabel("East coordinate from array centre (m)", fontsize=14)
    plt.ylabel("North coordiante from array centre (m)", fontsize=14)
    plt.title(
        f"{context.sched_start_utc}\n"
        + rf"Max. baseline $\approx$ {max_baseline:.0f} m"
    )
    plt.minorticks_on()
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.grid(which="minor", ls=":")
    plt.savefig(f"{context.obs_id}_array_layout.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
