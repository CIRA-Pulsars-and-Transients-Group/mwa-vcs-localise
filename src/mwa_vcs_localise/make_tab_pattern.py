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
from .utils import (
    MWA_LOCATION,
    sky_area,
    find_max_baseline,
    plot_array_layout,
    plot_primary_beam,
    plot_tied_array_beam,
    plot_tied_array_beam_1look_2freq,
)
from .array_factor import (
    extractWorkingTilePositions,
    calcGeometricDelays,
    calcArrayFactorPower,
)
from .primary_beam import getPrimaryBeamPower
from .stats import seekat, snr_reader


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
        help="Observing frequency in Hz.",
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
        You can instead define a box around the look-direction to simulate using
        the --gridbox option.
        """,
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
        help="Whether to produce plots of beam patterns.",
    )
    parser.add_argument(
        "--seekat",
        action="store_true",
        help="Whether to estimate localization based on seeKAT method.",
    )
    parser.add_argument(
        "--detfile",
        type=str,
        help="Path to a CSV, at least containing columns labeled as ra, dec, snr",
        default=None,
    )
    parser.add_argument(
        "--truth",
        type=str,
        help="Known true position of the target source (format: 'hh:mm:ss_dd:mm:ss').",
        default=None,
    )

    args = parser.parse_args()
    if len(args.freq) > 10:
        print("Cannot use more than 10 frequencies at a time, please adjust input.")
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
    width = ((sol.value / freqs) / max_baseline) * u.rad
    print(f"... maximum baseline (m): {max_baseline}")
    print(f"... beam width ~ lambda/D (arcmin): {width.to(u.arcminute).value}")
    time = Time(args.time, format="isot", scale="utc")
    altaz_frame = AltAz(location=MWA_LOCATION, obstime=time)

    if args.plot:
        print("Plotting array layout...")
        plot_array_layout(context)

    # Create the astrometric quantity for the beamformed target direction
    print("Creating look-direction vector...")
    if args.detfile:
        look_positions = snr_reader(args.detfile)[0]
    else:
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
        print(f"... sky box limits = ra{box_ra} dec{box_dec}")
        sky_area_sr = sky_area(box_ra, box_dec)
        print(f"... sky area = {sky_area_sr} = {sky_area_sr.to(u.deg**2)}")

        # Dump the grid to disk - deprecated, to be removed
        # np.savez("grid", grid_ra, grid_dec)

        target_positions = SkyCoord(
            grid_ra,
            grid_dec,
            frame="icrs",
            unit=("deg", "deg"),
        )
        print(f"... target positions array shape, (nRA, nDec) = {n_ra, n_dec}")
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

    # Compute and store the primary beam map, if requested
    pbp_freq = []
    for j, freq in enumerate(freqs):
        if args.nopb:
            pbp_freq.append(None)
        else:
            # Compute the primary beam zenith-normalised power.
            print(f"Computing primary beam power at frequency = {freq} Hz...")
            t0 = timer.time()
            pbp = getPrimaryBeamPower(
                context,
                freq,
                target_positions_altaz.alt.rad,
                target_positions_altaz.az.rad,
                stokes="I",
            )["I"].reshape(grid_ra.shape)
            pbp_freq.append(pbp)
            print(f"... primary beam max. in-field power = {pbp.max():.3f}")
            t1 = timer.time()
            print(f"... took {t1-t0} seconds")
    pbp_freq = np.array(pbp_freq)

    if not args.nopb:
        print("Plotting primary beam map...")
        plot_primary_beam(
            context,
            pbp_freq[0, ...],
            grid_ra,
            grid_dec,
            [0.05, 0.1, 0.25, 0.5, 0.8, 1],
            target=look_positions[0],
        )

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
            print(f"Processing tied-array beam at frequency = {freq} Hz")
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

            # Finally, estimate the zenith-normalised tied-array beam power.
            if args.nopb:
                tabp = afp
            else:
                tabp = afp * pbp_freq[j, ...]
            tabp_freq.append(tabp)
            afp_freq.append(afp)
        tabp_look.append(tabp_freq)
        afp_look.append(afp_freq)

    tabp_look = np.array(tabp_look)
    afp_look = np.array(afp_look)

    # tabp_look has shape: (nlook, nfreq, ndec, nra)
    # print(f"mean = {tabp_look.mean(axis=(-1,-2))/sky_area_sr.value}")
    # print(f"variance = {tabp_look.var(axis=(-1,-2))/sky_area_sr.value}")
    # print(tabp_look[0][0].shape)
    # spatial_covar = np.cov(tabp_look[0][0].ravel(), tabp_look[1][0].ravel())
    # print(spatial_covar)

    # Dump the tab map to disk - deprecated, to be removed
    # np.save("tabp_look", tabp_look)

    if args.plot:
        ctr_levels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.8, 1]
        oname_suffix = ""
        if args.nopb:
            tab_cbar_label = "Array factor power"
            oname_suffix += "_nopb"
        else:
            tab_cbar_label = "Zenith-normalised tied-array beam power"
            oname_suffix += "_pb"

        if len(args.freq) > 1:
            oname_suffix += "_multifreq"

        print("Plotting tied-array beam map...")
        plot_tied_array_beam(
            context,
            tabp_look,
            grid_ra,
            grid_dec,
            ctr_levels,
            tab_cbar_label,
            oname_suffix,
        )
        plot_tied_array_beam_1look_2freq(
            context,
            tabp_look[0],
            grid_ra,
            grid_dec,
            freqs,
            tab_cbar_label,
            oname_suffix,
        )

    tt1 = timer.time()
    print(f"Done!! (Took {tt1-tt0} seconds.)\n")

    if args.seekat:
        if args.detfile != None:
            if args.truth != None:
                true_coords = SkyCoord(
                    args.truth,
                    frame="icrs",
                    unit=("hourangle", "deg"),
                )
            else:
                true_coords = None
            loc, cov = seekat(
                args.detfile,
                tabp_look,
                grid_ra,
                grid_dec,
                truth_coords=true_coords,
            )
            loc.savefig("localisation.png", dpi=300)
            cov.savefig("covariance.png", dpi=300)
        else:
            print("ERROR: No detection file provided.")


if __name__ == "__main__":
    main()
