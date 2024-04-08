#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
import time as timer
import argparse
import mwalib
import numpy as np
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
from astropy.constants import c as sol
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from .utils import (
    MWA_LOCATION,
    form_grid_positions,
    find_max_baseline,
    plot_array_layout,
)
from .array_factor import (
    extractWorkingTilePositions,
    calcGeometricDelays,
    calcArrayFactorPower,
)
from .primary_beam import getPrimaryBeamPower

plt.rcParams.update(
    {
        "font.family": "serif",
    }
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        dest="metafits",
        type=str,
        help="Metafits file of the associated observation.",
    )
    parser.add_argument(
        "-t", dest="time", type=str, help="UTC time of observation (format: ISOT)."
    )
    parser.add_argument(
        "-f",
        dest="freq",
        nargs="+",
        type=float,
        help="Observing frequency in Hz. Maximum of 5.",
    )
    parser.add_argument(
        "-L",
        dest="look",
        type=str,
        help="Look-direction sky position (format: 'hh:mm:ss_dd:mm:ss')",
    )
    parser.add_argument(
        "-P",
        dest="position",
        type=str,
        help="""Sky position to compute array factor given the look-direction
        (format: 'hh:mm:ss_dd:mm:ss').
        You may provide multiple sky positions to sample, separating them by
        a single <space>.
        You can define a box around the look-direction to simulate using the --gridbox option.
        If no argument provided, will sample the sky around position based on
        an estimate of the FWHM. """,
        default=None,
    )
    parser.add_argument(
        "--gridbox",
        type=str,
        help="""Coordinates (RA/Dec) defining the box to sample. 
        Format is a single string as follows: 'RA0 Dec0 RA1 Dec1 RAstep Decstep' in h:m:s d:m:s,
        where (RA0, Dec0) is one corner and (RA1, Dec1) is the opposite corner, and '*step' 
        is the grid pixel size in arcseconds.""",
        default=None,
    )
    parser.add_argument(
        "--nopb",
        action="store_true",
        help="Don't include the primary beam attenuation.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the TAB power for each provided point.",
    )

    args = parser.parse_args()
    if len(args.freq) > 5:
        print("Cannot use more than 5 frequencies at a time, please adjust input.")
        exit(1)
    freqs = np.array(args.freq)
    if args.gridbox:
        grid_box = args.gridbox.split(" ")[:-2]
        grid_step = args.gridbox.split(" ")[-2:]

    tt0 = timer.time()
    print("Preparing metadata...")
    # Collect meta information and setup configuration.
    context = mwalib.MetafitsContext(args.metafits)
    max_baseline, _, _ = find_max_baseline(context)
    tile_positions = extractWorkingTilePositions(context)
    fwhm = (1.22 * (sol.value / freqs) / max_baseline) * u.rad
    print(f"... maximum baseline (m): {max_baseline}")
    print(f"... beam fwhm (arcmin): {fwhm.to(u.arcminute).value}")
    time = Time(args.time, format="isot", scale="utc")
    altaz_frame = AltAz(location=MWA_LOCATION, obstime=time)
    print("... plotting array layout")
    plot_array_layout(context)

    # Create the astrometric quantity for the beamformed target direction
    print("Creating look-direction vector...")
    look_ras = []
    look_decs = []
    for p in args.look.split(" "):
        look_ras.append(p.split("_")[0])
        look_decs.append(p.split("_")[1])

    look_positions = SkyCoord(
        look_ras,
        look_decs,
        frame="icrs",
        unit=("hourangle", "deg"),
    )
    print("Converting to AltAz...")
    t0 = timer.time()
    look_positions_altaz = look_positions.transform_to(altaz_frame)
    t1 = timer.time()
    print(f"... took {t1-t0} seconds")

    # In principle, allow the user to provide N inputs separated by spaces, or just
    # ask for M pointings around the source
    target_ras = []
    target_decs = []

    t0 = timer.time()
    print(
        "Creating sky position vectors from highest frequency and first look-direction..."
    )
    if not args.position and args.gridbox:
        box = SkyCoord(
            [grid_box[0], grid_box[2]],
            [grid_box[1], grid_box[3]],
            frame="icrs",
            unit=("hourangle", "deg"),
        )
        grid_step_ra = grid_step[0] * u.arcsec
        grid_step_dec = grid_step[1] * u.arcsec
        spherical_offset = box[0].spherical_offsets_to(box[1])
        n_ra = int(np.abs(spherical_offset[0].to(u.arcsec) / grid_step_ra))
        n_dec = int(np.abs(spherical_offset[1].to(u.arcsec) / grid_step_dec))

        if box.ra[0] > 180 * u.deg:
            print("... will wrap at RA = 0h such that grid spans -180 to +180 deg")
            box_ra = box.ra.deg
            box_ra[box_ra >= 180] -= 360
            box_dec = box.dec.deg
        else:
            box_ra = box.ra.deg
            box_dec = box.dec.deg

        grid_ra, grid_dec = np.meshgrid(
            np.linspace(box_ra[0], box_ra[1], n_ra),
            np.linspace(box_dec[0], box_dec[1], n_dec),
        )
        np.savez("grid", grid_ra, grid_dec)
        target_positions = SkyCoord(
            grid_ra,
            grid_dec,
            frame="icrs",
            unit=("deg", "deg"),
        )
        print(f"... target positions array shape, (nRA, nDec) = {n_ra, n_dec}")

    elif not args.position and args.gridbox is None:
        target_positions = form_grid_positions(
            look_positions[0],
            # max_separation_arcsec=(min(fwhm / 2)).to(u.arcsecond).value,
            max_separation_arcsec=(0.5 * u.arcmin).to(u.arcsecond).value,
            nlayers=args.nlayers,
            overlap=True,
        )
    else:
        for p in args.position.split(" "):
            target_ras.append(p.split("_")[0])
            target_decs.append(p.split("_")[1])

        target_positions = SkyCoord(
            target_ras,
            target_decs,
            frame="icrs",
            unit=("hourangle", "deg"),
        )
    t1 = timer.time()
    print(f"... took {t1-t0} seconds")

    print("Converting to AltAz...")
    t0 = timer.time()
    target_positions_altaz = target_positions.transform_to(altaz_frame)
    t1 = timer.time()
    print(f"... took {t1-t0} seconds")

    tabp_look = []
    afp_look = []
    for i, lp in enumerate(look_positions_altaz):
        print(
            "Processing look-direction = "
            f"{look_positions[i].ra.to_string(u.hour)} "
            f"{look_positions[i].dec.to_string(u.degree, alwayssign=True)}"
        )
        tabp_freq = []
        afp_freq = []
        for j, freq in enumerate(freqs):
            print(f"Processing frequency = {freq} Hz")
            print("Computing array factors...")
            t0 = timer.time()
            # Compute the array factor (tied-array beam weighting factor).
            look_psi = calcGeometricDelays(
                tile_positions,
                freq,
                lp.alt.rad,
                lp.az.rad,
            )
            target_psi = calcGeometricDelays(
                tile_positions,
                freq,
                target_positions_altaz.alt.rad,
                target_positions_altaz.az.rad,
            )
            afp = calcArrayFactorPower(look_psi, target_psi)
            t1 = timer.time()
            print(f"... took {t1-t0} seconds")

            if not args.nopb:
                # Compute the primary beam zenith-normalised power.
                print("Computing primary beam power...")
                t0 = timer.time()
                pbp = getPrimaryBeamPower(
                    context,
                    freq,
                    target_positions_altaz.alt.rad,
                    target_positions_altaz.az.rad,
                )
                print(f"... primary beam max. in-field power = {pbp.max():.3f}")
                t1 = timer.time()
                print(f"... took {t1-t0} seconds")
            else:
                pbp = 1.0

            # Finally, estimate the zenith-normalised tied-array beam power.
            if isinstance(pbp, float):
                tabp = afp
            else:
                tabp = afp * pbp.reshape(afp.shape)
            tabp_freq.append(tabp)
            afp_freq.append(afp)
        tabp_look.append(tabp_freq)
        afp_look.append(afp_freq)

    tabp_look = np.array(tabp_look)
    afp_look = np.array(afp_look)

    if args.plot:
        if args.nopb:
            label = "Array factor power"
        else:
            label = "Zenith-normalised tied-array beam power"

        print("Plotting sky map...")
        product = np.sum(tabp_look.mean(axis=1), axis=0)
        np.save("tabp_look", tabp_look)
        fig = plt.figure()
        ax = fig.add_subplot()
        ctr_levels = [0.01, 0.1, 0.25, 0.5, 0.8, 1]
        cmap = plt.get_cmap("Greys")

        map_extent = [
            grid_ra.min(),
            grid_ra.max(),
            grid_dec.min(),
            grid_dec.max(),
        ]

        tab_map = ax.imshow(
            tabp_look.mean(axis=1)[0],
            aspect="auto",
            interpolation="none",
            # origin="lower",
            extent=map_extent,
            cmap=cmap,
            norm="log",
            vmin=min(ctr_levels),
            vmax=max(ctr_levels),
        )

        for ld in tabp_look.mean(axis=1):
            tab_ctr = ax.contour(
                grid_ra,
                grid_dec,
                ld,
                levels=ctr_levels[2:-1],
                cmap="plasma",
                norm="log",
            )

        ax.set_xlabel("Right Ascension (deg)", fontsize=14)
        ax.set_ylabel("Declination (deg)", fontsize=14)
        ax.tick_params(labelsize=12)

        cbar = plt.colorbar(
            tab_map,
            ticks=ctr_levels,
            format=mticker.ScalarFormatter(),
            extend="min",
        )
        cbar.add_lines(tab_ctr)
        cbar.set_label(fontsize=12, label=label)
        cbar.ax.tick_params(labelsize=11)

        oname_base = f"{context.obs_id}_tiedarray_beam"

        if args.nopb:
            oname_base += "_nopb"
        else:
            oname_base += "_pb"

        if len(args.freq) > 1:
            oname_base += "_multifreq"

        plt.savefig(f"{oname_base}.png", dpi=200, bbox_inches="tight")

    tt1 = timer.time()
    print(f"Done!! (Took {tt1-tt0} seconds.)\n")


if __name__ == "__main__":
    main()
