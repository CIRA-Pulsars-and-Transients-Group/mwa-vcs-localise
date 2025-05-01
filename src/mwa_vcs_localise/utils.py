#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
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


def sky_area(ra: np.ndarray, dec: np.ndarray) -> u.quantity:
    """Estimate the sky area given a list of RA and Dec. coordinates that
    inscribe some ~rectangle on the sky.

    Args:
        ra (np.ndarray): The array of RA coordinates describing the E-W extent of the box.
        dec (np.ndarray): The array of Dec. coordinates describing the N-S extent of the box.

    Returns:
        u.quantity: An estimated sky area, in steradians.
    """

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    # For a rectangular "box" on the sky...
    # sin(north-most latitude) - sin(south-most latitude)
    c1 = np.sin(dec_rad.max()) - np.sin(dec_rad.min())
    # (east-most longitude) - (west-most longitude), remembering RA increases to the east
    c2 = ra_rad.max() - ra_rad.min()

    cap_area = c1 * c2

    return cap_area * u.sr


def find_characteristic_baseline(
    context: MetafitsContext, hdi_prob: float = 0.75
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """From the observation metadata, compute the tile effective and
    maximum baselines, as well as the baseline distribution.

    Args:
        context (MetafitsContext): A mwalib.MetafitsContext object that contains the
            array configuration and delay settings.
        hdi_prob (float, optional): Fraction of baselines to be included for the
            highest-density interval. Defaults to 0.75.

    Returns:
        tuple[float, np.ndarray, float, np.ndarray]: A tuple containing:
            (1) The effective (modal) baseline,
            (2) The highest-density interval (there may be more than one interval),
            (3) The maximum baseline, and
            (4) The baseline distribution.
    """
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
    """Plot the tile position layout.

    Args:
        context (MetafitsContext): A mwalib.MetafitsContext object that contains the
            array configuration and delay settings.
        ew_limits (list, optional): The E-W limits, relative to the array centre
            (in metres) to plot. Defaults to [-410, 410].
        ns_limits (list, optional): The N-S limits, relative to the array centre
            (in metres) to plot. Defaults to [50, 600].
    """
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
    """Plot the baseline distribution and indicate the highest-density interval(s).

    Args:
        context (MetafitsContext): A mwalib.MetafitsContext object that contains the
            array configuration and delay settings.
    """
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
    target: SkyCoord | None = None,
) -> None:
    """Plot the primary beam response across the gridded sky area.

    Args:
        context (MetafitsContext): A mwalib.MetafitsContext object that contains the
            array configuration and delay settings.
        pb (np.ndarray): The 2D primary beam map.
        gra (np.ndarray): The 2-D mesh grid in R.A. that defines the sky area of interest.
        gdec (np.ndarray): The 2-D mesh grid in Dec. that defines the sky area of interest.
        levels (list): Contour levels to plot, in units of primary beam power (0-1).
        target (SkyCoord | None, optional): A target position to highlight, if desired. Defaults to None.
    """

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
    label: str | None = None,
    oname_suffix: str | None = None,
) -> None:
    """Plot the tied-array beam pattern response across the gridded sky area.

    Args:
        context (MetafitsContext): A mwalib.MetafitsContext object that contains the
            array configuration and delay settings.
        tab (np.ndarray): The 2D tied-array beam map.
        gra (np.ndarray): The 2-D mesh grid in R.A. that defines the sky area of interest.
        gdec (np.ndarray): The 2-D mesh grid in Dec. that defines the sky area of interest.
        levels (list): Contour levels to plot, in units of tied-array beam power (0-1).
        label (str | None, optional): Label to describe the colorbar. Defaults to None (i.e., no label).
        oname_suffix (str | None, optional): A suffix to append to the end of the saved figure file.
            Defaults to None (i.e., figure named f"{context.obsid}_tiedarray_beam.png").
    """

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


def __plot_tab_centres_and_contours(
    beam_cen_coords,
    tabp,
    grid_ra,
    grid_dec,
    label,
    contours=True,
) -> None:
    """
    Making a plot of the beam and contours for looks, with the beam
    centres marked. Mostly for debugging.
    """

    tabp_sum = np.sum(tabp, axis=0)

    map_extent = [grid_ra.min(), grid_ra.max(), grid_dec.min(), grid_dec.max()]

    aspect = "equal"

    cmap = cm.get_sub_cmap(cm.cosmic, 0.1, 0.9)
    cmap.set_bad("red")
    contour_cmap = cm.get_sub_cmap(cm.cosmic_r, 0.1, 0.9)
    cmapnorm_sum = colors.Normalize(vmin=1e-5, vmax=0.1, clip=True)
    cmapnorm_indiv = colors.Normalize(vmin=1e-5, vmax=0.05, clip=True)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1_img = ax1.imshow(
        tabp_sum, aspect=aspect, extent=map_extent, cmap=cmap, norm=cmapnorm_sum
    )

    ax1.plot(
        beam_cen_coords.ra.deg,
        beam_cen_coords.dec.deg,
        "Dy",
        mec="k",
        ms=5,
        label="Beam centers",
    )

    if contours:
        for ls, look in enumerate(tabp):
            ax1.contour(
                look,
                origin="image",
                extent=map_extent,
                cmap=contour_cmap,
                norm=cmapnorm_indiv,
                linewidths=0.5,
            )

    ax1.legend(fontsize=18, loc=2)
    ax1.set_xlabel("R.A. (ICRS)", fontsize=18, ha="center")
    ax1.set_ylabel("Dec. (ICRS)", fontsize=18, ha="center")
    ax1.minorticks_on()
    ax1.tick_params(axis="both", which="major", labelsize=18)
    ax1.tick_params(axis="both", which="major", length=9)
    ax1.tick_params(axis="both", which="minor", length=4.5)
    ax1.tick_params(axis="both", which="both", direction="out", right=True, top=True)

    cbar = fig.colorbar(
        ax1_img,
        ax=fig.axes,
        shrink=1,
        orientation="horizontal",
        location="top",
        aspect=30,
        pad=0.02,
    )
    cbar.ax.set_title(label, fontsize=18, ha="center")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(direction="in", length=5, bottom=True, top=True)
    cbar.ax.xaxis.set_tick_params(labelsize=18)

    plt.savefig("tabs_with_centres.png", bbox_inches="tight")
