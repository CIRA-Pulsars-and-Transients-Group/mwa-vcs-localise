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
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import add_scalebar
import astropy.units as u
from mwalib import MetafitsContext, Pol
import arviz as az

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


def generate_wcs_grid(
    grid_ctr: SkyCoord,
    arcsec_per_pixel: float = 36.0,
    image_size: int | tuple[int] = 1000,
) -> tuple[np.ndarray, np.ndarray, WCS]:
    """Create a WCS frame and grid provided a central position,
    nominal pixel size and "image size" (i.e., grid size).

    Args:
        grid_ctr (SkyCoord): The centre coordinate for the image/grid.
        arcsec_per_pixel (float, optional): The number of arcseconds per pixel. Defaults to 36.0.
        image_size (int | tuple[int], optional): The image size, in pixels. Defaults to 1000.

    Returns:
        7tuple[np.ndarray, np.ndarray, WCS]: The grid in RA, Dec and the WCS object which can
        be used elsewhere to ensure consistent sky coordinate navigation and projections
    """
    # Set image size in pixels
    if isinstance(image_size, (tuple, list)):
        naxis1 = image_size[0]
        naxis2 = image_size[1]
    else:
        naxis1 = image_size
        naxis2 = image_size

    # Set central pixel as center of grid
    crpix1 = naxis1 / 2
    crpix2 = naxis2 / 2

    # Set the pixel scale
    pixel_scale = arcsec_per_pixel / 3600  # deg/pixel

    wcs_dict = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": grid_ctr.ra.deg,
        "CRVAL2": grid_ctr.dec.deg,
        "CRPIX1": crpix1,
        "CRPIX2": crpix2,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CDELT1": -pixel_scale,
        "CDELT2": pixel_scale,
        "NAXIS1": naxis1,
        "NAXIS2": naxis2,
    }
    wcs = WCS(wcs_dict)

    # Generate a grid in pixel space
    gx, gy = np.meshgrid(np.arange(naxis1), np.arange(naxis2))
    pixel_coords = np.column_stack([gx.ravel(), gy.ravel()])

    # Convert pixel coordinates to "world" coordinates
    world_coords = wcs.wcs_pix2world(pixel_coords, 0)
    grid_ra = world_coords[:, 0].reshape(gy.shape)
    grid_dec = world_coords[:, 1].reshape(gy.shape)

    return grid_ra, grid_dec, wcs


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
    context: MetafitsContext,
    hdi_prob: float = 0.9,
    extra_tile_flags: list[str] | None = None,
    exclude_flagged: bool = True,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """From the observation metadata, compute the tile effective and
    maximum baselines, as well as the baseline distribution.

    Args:
        context (MetafitsContext): A mwalib.MetafitsContext object that contains the
            array configuration and delay settings.
        hdi_prob (float, optional): Fraction of baselines to be included for the
            highest-density interval. Defaults to 0.9.
        extra_tile_flags (list[str] | None, optional): A list of additional
        tile names to flag as bad. Defaults to None.
        exclude_flagged (bool, optional): Whether to exclude flagged tiles
            from the baseline distribution.
    Returns:
        tuple[float, float, np.ndarray, np.ndarray]: A tuple containing:
            (1) The baseline mode (i.e., the most common baseline length),
            (2) The maximum baseline,
            (3) The highest-density interval, and
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
    if extra_tile_flags is not None:
        itile = 0
        for rf in context.rf_inputs:
            if rf.pol != Pol.X:
                continue
            if rf.tile_name in extra_tile_flags or str(rf.tile_id) in extra_tile_flags:
                tile_flags[itile] = True
            itile += 1

    if exclude_flagged:
        tile_positions = np.delete(
            tile_positions,
            np.where(tile_flags & True),
            axis=0,
        )

    dist = cdist(tile_positions, tile_positions)
    dist = np.delete(dist.flatten(), np.where(dist.flatten() <= 0.01))  # remove autos
    max_dist = np.max(dist) * u.m
    distances = dist * u.m

    # use a KDE approach to estimate the mode of the baseline distribution
    grid, density = az.kde(dist)
    dist_mode = grid[np.argmax(density)] * u.m
    dist_hdi = np.asarray(az.hdi(dist, hdi_prob=hdi_prob, multimodal=False)) * u.m

    return dist_mode, max_dist, dist_hdi, distances


def plot_array_layout(
    context: MetafitsContext,
    ew_limits: list | None = None,
    ns_limits: list | None = None,
    extra_tile_flags: list[str] | None = None,
    show_flagged_tiles: bool = True,
) -> None:
    """Plot the tile position layout.

    Args:
        context (MetafitsContext): A mwalib.MetafitsContext object that contains the
            array configuration and delay settings.
        ew_limits (list, optional): The E-W limits, relative to the array centre
            (in metres) to plot. Defaults to None.
        ns_limits (list, optional): The N-S limits, relative to the array centre
            (in metres) to plot. Defaults to None.
        show_flagged_tiles (bool): Plot the flagged tiles in a different colour.
            Default: True.
    """
    tile_positions = np.array(
        [
            np.array([rf.east_m, rf.north_m, rf.height_m])
            for rf in context.rf_inputs
            if rf.pol == Pol.X
        ]
    )
    tile_flags = np.array([rf.flagged for rf in context.rf_inputs if rf.pol == Pol.X])
    if extra_tile_flags is not None:
        itile = 0
        for rf in context.rf_inputs:
            if rf.pol != Pol.X:
                continue
            if rf.tile_name in extra_tile_flags or str(rf.tile_id) in extra_tile_flags:
                tile_flags[itile] = True
            itile += 1

    _, max_baseline, hdi_baseline, _ = find_characteristic_baseline(context)
    eff_baseline = np.max(hdi_baseline) * u.m

    okay_tiles_n = np.ma.masked_array(tile_positions[:, 1], mask=tile_flags)
    okay_tiles_e = np.ma.masked_array(tile_positions[:, 0], mask=tile_flags)
    bad_tiles_n = np.ma.masked_array(tile_positions[:, 1], mask=~tile_flags)
    bad_tiles_e = np.ma.masked_array(tile_positions[:, 0], mask=~tile_flags)

    num_ok_tiles = (~tile_flags).sum()
    num_bad_tiles = (tile_flags).sum()

    fig = plt.figure(figsize=(8, 6))
    fig.add_subplot()

    if show_flagged_tiles:
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
    else:
        plt.scatter(
            tile_positions[:, 1],
            tile_positions[:, 0],
            zorder=1000,
            s=10,
            marker="x",
            color="k",
            label=f"All tiles ({len(tile_flags)})",
        )

    if ew_limits:
        plt.xlim(ew_limits)
    if ns_limits:
        plt.ylim(ns_limits)
    plt.legend(fontsize=12)
    plt.xlabel("East coordinate from array centre (m)", fontsize=14)
    plt.ylabel("North coordiante from array centre (m)", fontsize=14)
    plt.title(
        f"Observation ID: {context.obs_id}  ({context.sched_start_utc})\n"
        + rf"Max. baseline $\approx$ {max_baseline*u.m:.0f}  "
        + rf"Characteristic baseline $\approx$ {eff_baseline:.0f}"
    )
    plt.minorticks_on()
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.grid(which="minor", ls=":")
    plt.savefig(f"{context.obs_id}_array_layout.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_distribution(
    context: MetafitsContext,
    extra_tile_flags: list[str] | None = None,
    show_flagged_tiles: bool = True,
) -> None:
    """Plot the baseline distribution and indicate the highest-density interval(s).

    Args:
        context (MetafitsContext): A mwalib.MetafitsContext object that contains the
            array configuration and delay settings.
        extra_tile_flags (list[str] | None, optional): A list of additional
            tile names to flag as bad. Defaults to None.
        show_flagged_tiles (bool): Plot the flagged tiles in a different colour.
            Default: True.
    """
    _, max_baseline, hdi_baseline, baselines = find_characteristic_baseline(
        context, extra_tile_flags=extra_tile_flags
    )
    eff_baseline = np.max(hdi_baseline)

    tile_flags = np.array([rf.flagged for rf in context.rf_inputs if rf.pol == Pol.X])
    if extra_tile_flags is not None:
        itile = 0
        for rf in context.rf_inputs:
            if rf.pol != Pol.X:
                continue
            if rf.tile_name in extra_tile_flags or str(rf.tile_id) in extra_tile_flags:
                tile_flags[itile] = True
            itile += 1

    num_ok_tiles = (~tile_flags).sum()
    num_bad_tiles = (tile_flags).sum()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    ax.hist([b.value for b in baselines], bins=np.arange(0, max_baseline.value, 10))
    ymax = max(ax.get_ylim())

    if len(np.shape(hdi_baseline)) > 1:
        for i in list(hdi_baseline):
            ax.fill_between(i.value, 0, ymax, color="0.8", alpha=0.5)
    else:
        ax.fill_between(
            [h.value for h in hdi_baseline],
            0,
            ymax,
            color="0.8",
            alpha=0.5,
        )
    ax.axvline(eff_baseline.value, ls=":", color="k")
    ax.text(
        x=0.95,
        y=0.95,
        s=f"Number of baselines = {len(baselines)}\n"
        + f"Number of 'good' tiles = {num_ok_tiles}\n"
        + f"Number of flagged tiles = {num_bad_tiles}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=12,
    )
    plt.xlim(0, None)
    plt.ylim(None, ymax)
    plt.xlabel("Baseline length (m)", fontsize=14)
    plt.ylabel("Frequency of baseline length", fontsize=14)
    plt.title(
        f"Observation ID: {context.obs_id}  ({context.sched_start_utc})\n"
        + rf"Max. baseline $\approx$ {max_baseline*u.m:.0f}  "
        + rf"Characteristic baseline $\approx$ {eff_baseline:.0f}"
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
    wcs: WCS | None,
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
        wcs: (WCS | None): The astropy WCS object defining the world coordinate system.
        target (SkyCoord | None, optional): A target position to highlight, if desired. Defaults to None.
    """

    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    pb_map = ax.imshow(
        pb,
        aspect="auto",
        interpolation="none",
        cmap=cm.cosmic_r,
        norm="log",
        vmin=min(levels),
        vmax=max(levels),
    )
    pb_ctr = ax.contour(
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
    ax.set_xlabel("Right Ascension", fontsize=14)
    ax.set_ylabel("Declination", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(ls=":")

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
    wcs: WCS | None,
    scale_arcmin: float | None = 1.0,
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
        wcs: (WCS | None): The astropy WCS object defining the world coordinate system.
        scale_arcmin: (float | None): The scale, in arcmin, to show as a scale bar on the plot. Default is 1'.
        label (str | None, optional): Label to describe the colorbar. Defaults to None (i.e., no label).
        oname_suffix (str | None, optional): A suffix to append to the end of the saved figure file.
            Defaults to None (i.e., figure named f"{context.obsid}_tiedarray_beam.png").
    """

    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=wcs)

    tab_map = ax.imshow(
        tab.mean(axis=1)[0],
        aspect="auto",
        interpolation="none",
        origin="lower",
        cmap=cm.sapphire_r,
        norm="log",
        vmin=min(levels),
        vmax=max(levels),
    )

    for ld in tab.mean(axis=1):
        tab_ctr = ax.contour(
            ld,
            levels=levels[1:-1],
            cmap="plasma",
            norm="log",
        )

    ax.set_xlabel("Right Ascension", fontsize=14)
    ax.set_ylabel("Declination", fontsize=14)
    ax.relim()
    ax.autoscale_view()
    ax.tick_params(labelsize=12)
    ax.grid(ls=":")

    # Add a scale bar
    if scale_arcmin:
        add_scalebar(
            ax, scale_arcmin * u.arcmin, label=f"{scale_arcmin}'", color="black"
        )

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
