########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import arviz as az
import astropy.units as u
import cmasher as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.visualization.wcsaxes import add_scalebar
from astropy.wcs import WCS
from mwalib import MetafitsContext, Pol
from scipy.spatial.distance import cdist

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
    """Create a tangent-plane WCS and corresponding RA/Dec coordinate grids.

    :param grid_ctr: Central sky coordinate for the grid.
    :type grid_ctr: SkyCoord
    :param arcsec_per_pixel: Pixel scale in arcseconds.
    :type arcsec_per_pixel: float
    :param image_size: Grid size in pixels as ``N`` or ``(Nx, Ny)``.
    :type image_size: int | tuple[int]
    :returns: RA grid, Dec grid, and WCS object.
    :rtype: tuple[np.ndarray, np.ndarray, WCS]
    """
    # Set image size in pixels
    if isinstance(image_size, (tuple, list)):
        naxis1 = image_size[0]
        naxis2 = image_size[1]
    else:
        naxis1 = image_size
        naxis2 = image_size

    # Set central pixel as centre of grid
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
    """Estimate sky area enclosed by RA/Dec extrema, in steradians.

    :param ra: RA samples describing east-west extent in degrees.
    :type ra: np.ndarray
    :param dec: Dec samples describing north-south extent in degrees.
    :type dec: np.ndarray
    :returns: Estimated area on the sphere.
    :rtype: u.quantity
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
) -> tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    """Compute baseline distribution summary statistics from observation metadata.

    :param context: MWALIB metadata containing tile positions and flags.
    :type context: MetafitsContext
    :param hdi_prob: Probability mass for highest-density interval estimate.
    :type hdi_prob: float
    :param extra_tile_flags: Additional tile names or IDs to flag as bad.
    :type extra_tile_flags: list[str] | None
    :param exclude_flagged: If True, ignore flagged tiles in baseline estimates.
    :type exclude_flagged: bool
    :returns: Baseline mode, maximum baseline, HDI interval, and baseline
        sample distribution.
    :rtype: tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]
    """
    tile_positions = np.array(
        [
            np.array([rf.east_m, rf.north_m, rf.height_m])
            for rf in context.rf_inputs
            if rf.pol == Pol.X
        ]
    )
    tile_flags = np.array(
        [rf.flagged for rf in context.rf_inputs if rf.pol == Pol.X]
    )
    if extra_tile_flags is not None:
        itile = 0
        for rf in context.rf_inputs:
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
            tile_positions,
            np.where(tile_flags & True),
            axis=0,
        )

    dist = cdist(tile_positions, tile_positions)
    dist = np.delete(
        dist.flatten(), np.where(dist.flatten() <= 0.01)
    )  # remove autos
    max_dist = np.max(dist) * u.m
    distances = dist * u.m

    # use a KDE approach to estimate the mode of the baseline distribution
    grid, density, _ = az.kde(dist)
    dist_mode = grid[np.argmax(density)] * u.m
    dist_hdi = np.asarray(az.hdi(dist, prob=hdi_prob, method="nearest")) * u.m

    return dist_mode, max_dist, dist_hdi, distances


def plot_array_layout(
    context: MetafitsContext,
    ew_limits: list | None = None,
    ns_limits: list | None = None,
    extra_tile_flags: list[str] | None = None,
    show_flagged_tiles: bool = True,
) -> None:
    """Plot the array tile layout in local east-north coordinates.

    :param context: MWALIB metadata containing tile positions and flags.
    :type context: MetafitsContext
    :param ew_limits: Optional east-west axis limits in metres.
    :type ew_limits: list | None
    :param ns_limits: Optional north-south axis limits in metres.
    :type ns_limits: list | None
    :param extra_tile_flags: Additional tile names or IDs to flag as bad.
    :type extra_tile_flags: list[str] | None
    :param show_flagged_tiles: If True, display flagged tiles separately.
    :type show_flagged_tiles: bool
    :returns: None
    :rtype: None
    """
    tile_positions = np.array(
        [
            np.array([rf.east_m, rf.north_m, rf.height_m])
            for rf in context.rf_inputs
            if rf.pol == Pol.X
        ]
    )
    tile_flags = np.array(
        [rf.flagged for rf in context.rf_inputs if rf.pol == Pol.X]
    )
    if extra_tile_flags is not None:
        itile = 0
        for rf in context.rf_inputs:
            if rf.pol != Pol.X:
                continue
            if (
                rf.tile_name in extra_tile_flags
                or str(rf.tile_id) in extra_tile_flags
            ):
                tile_flags[itile] = True
            itile += 1

    _, max_baseline, hdi_baseline, _ = find_characteristic_baseline(
        context,
        exclude_flagged=show_flagged_tiles,
    )
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
        + rf"Max. baseline $\approx$ {max_baseline * u.m:.0f}  "
        + rf"Characteristic baseline $\approx$ {eff_baseline:.0f}"
    )
    plt.minorticks_on()
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.grid(which="minor", ls=":")
    plt.savefig(
        f"{context.obs_id}_array_layout.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def plot_baseline_distribution(
    context: MetafitsContext,
    extra_tile_flags: list[str] | None = None,
    show_flagged_tiles: bool = True,
) -> None:
    """Plot baseline length histogram and highlight HDI regions.

    :param context: MWALIB metadata containing tile positions and flags.
    :type context: MetafitsContext
    :param extra_tile_flags: Additional tile names or IDs to flag as bad.
    :type extra_tile_flags: list[str] | None
    :param show_flagged_tiles: If True, include flagged tile state in summary.
    :type show_flagged_tiles: bool
    :returns: None
    :rtype: None
    """
    _, max_baseline, hdi_baseline, baselines = find_characteristic_baseline(
        context,
        extra_tile_flags=extra_tile_flags,
        exclude_flagged=show_flagged_tiles,
    )
    eff_baseline = np.max(hdi_baseline)

    tile_flags = np.array(
        [rf.flagged for rf in context.rf_inputs if rf.pol == Pol.X]
    )
    if extra_tile_flags is not None:
        itile = 0
        for rf in context.rf_inputs:
            if rf.pol != Pol.X:
                continue
            if (
                rf.tile_name in extra_tile_flags
                or str(rf.tile_id) in extra_tile_flags
            ):
                tile_flags[itile] = True
            itile += 1

    num_ok_tiles = (~tile_flags).sum()
    num_bad_tiles = (tile_flags).sum()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    ax.hist(
        [b.value for b in baselines], bins=np.arange(0, max_baseline.value, 10)
    )
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
        + rf"Max. baseline $\approx$ {max_baseline * u.m:.0f}  "
        + rf"Characteristic baseline $\approx$ {eff_baseline:.0f}"
    )
    plt.minorticks_on()
    plt.tick_params(labelsize=12)
    plt.savefig(
        f"{context.obs_id}_baseline_dist.png", dpi=200, bbox_inches="tight"
    )
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
    """Plot primary-beam power and contour levels over the sampled sky grid.

    :param context: MWALIB metadata containing observation identifiers.
    :type context: MetafitsContext
    :param pb: Two-dimensional primary-beam power map.
    :type pb: np.ndarray
    :param gra: RA coordinate grid corresponding to ``pb``.
    :type gra: np.ndarray
    :param gdec: Dec coordinate grid corresponding to ``pb``.
    :type gdec: np.ndarray
    :param levels: Beam-power levels used for image scaling and contours.
    :type levels: list
    :param wcs: WCS projection for display.
    :type wcs: WCS | None
    :param target: Optional target coordinate to overlay.
    :type target: SkyCoord | None
    :returns: None
    :rtype: None
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
    """Plot tied-array beam response and contour levels over the sky grid.

    :param context: MWALIB metadata containing observation identifiers.
    :type context: MetafitsContext
    :param tab: Tied-array beam map cube.
    :type tab: np.ndarray
    :param gra: RA coordinate grid corresponding to ``tab``.
    :type gra: np.ndarray
    :param gdec: Dec coordinate grid corresponding to ``tab``.
    :type gdec: np.ndarray
    :param levels: Power levels used for image scaling and contours.
    :type levels: list
    :param wcs: WCS projection for display.
    :type wcs: WCS | None
    :param scale_arcmin: Scale-bar size in arcminutes.
    :type scale_arcmin: float | None
    :param label: Optional colourbar label text.
    :type label: str | None
    :param oname_suffix: Optional suffix added to output filename.
    :type oname_suffix: str | None
    :returns: None
    :rtype: None
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
            ax,
            scale_arcmin * u.arcmin,
            label=f"{scale_arcmin}'",
            color="black",
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
