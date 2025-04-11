#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmasher as cm

import numpy as np
from scipy.spatial.distance import cdist
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
from mwalib import MetafitsContext, Pol
import arviz as az
from arviz.plots.plot_utils import calculate_point_estimate


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


def find_characteristic_baseline(context: MetafitsContext, hdi_prob: float = 0.75):
    tile_positions = np.array(
        [
            np.array([rf.east_m, rf.north_m, rf.height_m])
            for rf in context.rf_inputs
            if rf.pol == Pol.X
        ]
    )
    tile_flags = np.array([rf.flagged for rf in context.rf_inputs if rf.pol == Pol.X])
    tile_positions = np.delete(tile_positions, np.where(tile_flags & True), axis=0)

    dist = cdist(tile_positions, tile_positions)
    dist = np.delete(dist, np.where(dist <= 0.01))  # remove autos

    # use a KDE approach to estimate the mode of the baseline distribution
    dist_mode = calculate_point_estimate("mode", dist)
    dist_hdi = az.hdi(dist, hdi_prob=hdi_prob, multimodal=True)

    return dist_mode, dist_hdi, max(dist), dist


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

    eff_b, _, max_b, _ = find_characteristic_baseline(context)

    okay_tiles_n = np.ma.masked_array(tile_positions[:, 1], mask=tile_flags)
    okay_tiles_e = np.ma.masked_array(tile_positions[:, 0], mask=tile_flags)
    bad_tiles_n = np.ma.masked_array(tile_positions[:, 1], mask=~tile_flags)
    bad_tiles_e = np.ma.masked_array(tile_positions[:, 0], mask=~tile_flags)

    num_ok_tiles = (~tile_flags).sum()
    num_bad_tiles = (tile_flags).sum()

    fig = plt.figure(figsize=(8, 6))
    fig.add_subplot()
    plt.scatter(
        okay_tiles_e,
        okay_tiles_n,
        zorder=1000,
        s=10,
        marker="x",
        color="k",
        label=f"'Good' tiles ({num_ok_tiles})",
    )
    plt.scatter(
        bad_tiles_e,
        bad_tiles_n,
        zorder=1000,
        s=10,
        marker="x",
        color="r",
        label=f"Flagged tiles ({num_bad_tiles})",
    )
    plt.xlim(ew_limits)
    plt.ylim(ns_limits)
    plt.legend(fontsize=10)
    plt.xlabel("East coordinate from array centre (m)", fontsize=14)
    plt.ylabel("North coordiante from array centre (m)", fontsize=14)
    plt.title(
        f"Observation ID: {context.obs_id}  ({context.sched_start_utc})\n"
        + rf"Max. baseline $\approx$ {max_b:.0f} m  "
        + rf"Characteristic baseline $\approx$ {eff_b:.0f} m"
    )
    plt.minorticks_on()
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.grid(which="minor", ls=":")
    plt.savefig(f"{context.obs_id}_array_layout.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_distribution(context: MetafitsContext) -> None:
    b_eff, hdi, b_max, b = find_characteristic_baseline(context)

    tile_flags = np.array([rf.flagged for rf in context.rf_inputs if rf.pol == Pol.X])
    num_ok_tiles = (~tile_flags).sum()
    num_bad_tiles = (tile_flags).sum()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    ax.hist(b, bins="auto")
    ymax = max(ax.get_ylim())
    for i in hdi:
        ax.fill_between(i, 0, ymax, color="0.8", alpha=0.5)
    ax.axvline(b_eff, ls=":", color="k")
    ax.text(
        x=0.95,
        y=0.95,
        s=f"Number of baselines = {len(b)}\n"
        + f"Number of 'good' tiles = {num_ok_tiles}\n"
        + f"Number of flagged tiles = {num_bad_tiles}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
    )
    plt.xlim(0, None)
    plt.ylim(None, ymax)
    plt.xlabel("Baseline length (m)", fontsize=14)
    plt.ylabel("Frequency of baseline length", fontsize=14)
    plt.title(
        f"Observation ID: {context.obs_id}  ({context.sched_start_utc})\n"
        + rf"Max. baseline $\approx$ {b_max:.0f} m  "
        + rf"Characteristic baseline $\approx$ {b_eff:.0f} m"
    )
    plt.minorticks_on()
    plt.tick_params(labelsize=12)
    plt.savefig(f"{context.obs_id}_baseline_dist.png", dpi=200, bbox_inches="tight")
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
        origin="lower",
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
