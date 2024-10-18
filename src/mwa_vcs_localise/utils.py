#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmasher as cm

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
from mwalib import MetafitsContext, Pol

# Plotting style/formats
plt.rcParams.update(
    {
        "font.family": "serif",
    }
)

# Define MWA location
MWA_CENTRE_LON = 116.67081524 * u.deg
MWA_CENTRE_LAT = -26.70331940 * u.deg
MWA_CENTRE_H = 377.8269 * u.m
MWA_CENTRE_CABLE_LEN = 0.0 * u.m

MWA_LOCATION = EarthLocation.from_geodetic(
    lon=MWA_CENTRE_LON, lat=MWA_CENTRE_LAT, height=MWA_CENTRE_H
)


def sky_area(ra, dec):

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    # For a rectangular "box" on the sky...
    # sin(north-most latitude) - sin(south-most latitude)
    c1 = np.sin(dec_rad.max()) - np.sin(dec_rad.min())
    # (east-most longitude) - (west-most longitude), remembering RA increases to the east
    c2 = ra_rad.max() - ra_rad.min()

    cap_area = c1 * c2

    return cap_area * u.sr


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
            for rf in context.rf_inputs
            if rf.Pol == Pol.X
        ]
    )
    tile_flags = np.array([rf.flagged for rf in context.rf_inputs if rf.Pol == Pol.X])
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


def plot_array_layout(
    context: MetafitsContext,
    ew_limits: list = [-410, 410],
    ns_limits: list = [50, 600],
) -> None:
    tile_positions = np.array(
        [
            np.array([rf.east_m, rf.north_m, rf.height_m])
            for rf in context.rf_inputs
            if rf.pol == Pol.X
        ]
    )
    tile_flags = np.array([rf.flagged for rf in context.rf_inputs if rf.pol == Pol.X])
    max_baseline = find_max_baseline(context)[0]

    okay_tiles_n = np.ma.masked_array(tile_positions[:, 1], mask=tile_flags)
    okay_tiles_e = np.ma.masked_array(tile_positions[:, 0], mask=tile_flags)
    bad_tiles_n = np.ma.masked_array(tile_positions[:, 1], mask=~tile_flags)
    bad_tiles_e = np.ma.masked_array(tile_positions[:, 0], mask=~tile_flags)

    fig = plt.figure(figsize=(8, 6))
    fig.add_subplot()
    plt.scatter(
        okay_tiles_e,
        okay_tiles_n,
        zorder=1000,
        s=10,
        marker="x",
        color="k",
    )
    plt.scatter(
        bad_tiles_e,
        bad_tiles_n,
        zorder=1000,
        s=10,
        marker="x",
        color="r",
    )
    plt.xlim(ew_limits)
    plt.ylim(ns_limits)
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


def plot_primary_beam(
    context: MetafitsContext,
    pb: np.ndarray,
    gra: np.ndarray,
    gdec: np.ndarray,
    levels: list,
    target: SkyCoord = None,
) -> None:

    map_extent = [
        gra.min(),
        gra.max(),
        gdec.min(),
        gdec.max(),
    ]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    pb_map = ax.imshow(
        pb,
        aspect="auto",
        interpolation="none",
        extent=map_extent,
        cmap=cm.cosmic_r,
        norm="log",
        vmin=min(levels),
        vmax=max(levels),
    )
    pb_ctr = ax.contour(
        gra,
        gdec,
        pb,
        levels=levels[1:-1],
        cmap="plasma",
        norm="log",
    )
    if target:
        ax.scatter(
            target.ra.deg,
            target.dec.deg,
            c="r",
            marker="x",
            zorder=100,
        )
    ax.set_xlabel("Right Ascension (deg)", fontsize=14)
    ax.set_ylabel("Declination (deg)", fontsize=14)
    ax.tick_params(labelsize=12)

    cbar = plt.colorbar(
        pb_map,
        ticks=levels,
        format=mticker.ScalarFormatter(),
        extend="min",
        pad=0.02,
    )
    cbar.add_lines(pb_ctr)
    cbar.set_label(fontsize=12, label="Zenith-normalised primary beam power")
    cbar.ax.tick_params(labelsize=11)

    plt.savefig(f"{context.obs_id}_pb.png", dpi=200, bbox_inches="tight")


def plot_tied_array_beam(
    context: MetafitsContext,
    tab: np.ndarray,
    gra: np.ndarray,
    gdec: np.ndarray,
    levels: list,
    label: str = None,
    oname_suffix: str = None,
) -> None:

    map_extent = [
        gra.min(),
        gra.max(),
        gdec.min(),
        gdec.max(),
    ]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    tab_map = ax.imshow(
        tab.mean(axis=1)[0],
        aspect="auto",
        interpolation="none",
        # origin="lower",
        extent=map_extent,
        cmap=cm.sapphire_r,
        norm="log",
        vmin=min(levels),
        vmax=max(levels),
    )
    for ld in tab.mean(axis=1):
        tab_ctr = ax.contour(
            gra,
            gdec,
            ld,
            levels=levels[1:-1],
            cmap="plasma",
            norm="log",
        )
    ax.set_xlabel("Right Ascension (deg)", fontsize=14)
    ax.set_ylabel("Declination (deg)", fontsize=14)
    ax.tick_params(labelsize=12)

    tab_map.cmap.set_under("white")
    cbar = plt.colorbar(
        tab_map,
        ticks=levels,
        format=mticker.ScalarFormatter(),
        extend="min",
        pad=0.02,
    )
    cbar.add_lines(tab_ctr)
    if label:
        cbar.set_label(fontsize=12, label=label)
    cbar.ax.tick_params(labelsize=11)

    oname_base = f"{context.obs_id}_tiedarray_beam"
    if oname_suffix:
        oname_base += oname_suffix

    plt.savefig(f"{oname_base}.png", dpi=200, bbox_inches="tight")


def plot_tied_array_beam_1look_2freq(
    context: MetafitsContext,
    tab: np.ndarray,
    gra: np.ndarray,
    gdec: np.ndarray,
    freqs: list,
    label: str = None,
    oname_suffix: str = None,
) -> None:

    map_extent = [
        gra.min(),
        gra.max(),
        gdec.min(),
        gdec.max(),
    ]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    lines = []
    labels = []
    for i, (tabf, col) in enumerate(zip(tab, ["red", "blue"])):
        tab_ctr = ax.contour(
            gra,
            gdec,
            tabf,
            levels=[0.1, 0.5],
            colors=col,
            linestyles=["solid", "dotted"],
            norm="log",
        )
        lines.append(tab_ctr.legend_elements()[0][0])
        labels.append(f"{freqs[i]/1e6:.2f} MHz")
    ax.set_xlabel("Right Ascension (deg)", fontsize=14)
    ax.set_ylabel("Declination (deg)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(lines, labels)

    ax.set_ylim(-58, -52)
    ax.set_xlim(42, 48)

    oname_base = f"{context.obs_id}_tiedarray_beam_2freq"
    if oname_suffix:
        oname_base += oname_suffix

    plt.savefig(f"{oname_base}.png", dpi=200, bbox_inches="tight")
