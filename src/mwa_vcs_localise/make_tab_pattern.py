########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
import argparse
import sys
import time as timer

import astropy.constants as c
import astropy.units as u
import mwalib
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from .array_factor import (
    calc_array_factor_power,
    calc_geometric_delays,
    extract_working_tile_positions,
)
from .primary_beam import get_primary_beam_power
from .stats import localise, snr_reader
from .utils import (
    MWA_LOCATION,
    find_characteristic_baseline,
    generate_wcs_grid,
    plot_array_layout,
    plot_baseline_distribution,
    plot_primary_beam,
    plot_tied_array_beam,
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
        "-t",
        dest="time",
        type=str,
        help="UTC time of observation (format: ISOT).",
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
        "--use-wcs",
        action="store_true",
        help="Use WCS to define a grid around the central point.",
    )
    parser.add_argument(
        "--wcs-grid-size",
        help="""The WCS grid size, in pixels, the be created. The centre of the grid
          is either the provided 'look-direction' (-L option) or the first entry
          in the provided detection file (--detfile option).""",
        nargs=2,
        type=int,
        default=(1024, 1024),
    )
    parser.add_argument(
        "--wcs-pixel-size",
        help="""The size of each pixel in the WCS grid, in arcseconds""",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--nopb",
        action="store_true",
        help="DO NOT include the primary beam attenuation.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Produce diagnostic and result plots. Otherwise, just text is printed to stdout.",
    )
    parser.add_argument(
        "--localise",
        action="store_true",
        help="Localise and report results.",
    )
    parser.add_argument(
        "--detfile",
        type=str,
        help="Path to a CSV containing the header 'ra,dec,snr' and corresponding rows per detection.",
        default=None,
    )
    parser.add_argument(
        "--truth",
        type=str,
        help="Known true position of the target source (format: 'hh:mm:ss ±dd:mm:ss').",
        default=None,
    )
    parser.add_argument(
        "--regularise",
        type=str,
        help="Type of regularisation function to use.",
        choices=["none", "tab", "gaussian"],
        default="tab",
    )
    parser.add_argument(
        "--zoom",
        help="Create a figure inset zoomed on best-fit region.",
        action="store_true",
    )
    parser.add_argument(
        "--tile-flags",
        type=str,
        help="A comma-separated list of tile names or IDs to flag.",
        default=None,
    )
    parser.add_argument(
        "--no-tile-flags",
        help="Do not remove flagged tile positions when generating tied-array beam pattern.",
        action="store_true",
    )

    args = parser.parse_args()
    if len(args.freq) > 10:
        print(
            "Cannot use more than 10 frequencies at a time, please adjust input."
        )
        sys.exit(1)
    freqs = np.array([f for f in args.freq]) * u.Hz

    if args.regularise == "none":
        regularisation_fn = None
    else:
        regularisation_fn = args.regularise
    print(f"Regularisation function requested: {regularisation_fn}")

    tt0 = timer.time()
    print("Preparing metadata...")
    # Collect meta information and setup configuration.
    context = mwalib.MetafitsContext(args.metafits)

    # Examine the array layout, collect tile positions and baseline information
    density_interval_prob = 0.90
    char_baseline, max_baseline, hdi_baseline, baselines = (
        find_characteristic_baseline(
            context,
            hdi_prob=density_interval_prob,
            extra_tile_flags=args.tile_flags,
            exclude_flagged=(not args.no_tile_flags),
        )
    )
    eff_baseline = np.max(hdi_baseline)
    tile_positions, num_good, num_flagged = extract_working_tile_positions(
        context,
        extra_tile_flags=args.tile_flags,
        exclude_flagged=(not args.no_tile_flags),
    )
    num_tiles = num_good + num_flagged
    print(f"... number of tiles: {num_tiles}")
    print(f"... number of unflagged tiles: {num_good}")
    print(f"... number of baselines: {len(baselines)}")
    print(f"Maximum baseline, Bmax = {max_baseline:g}")
    print(f"Approx. mode of baselines = {char_baseline:g}")
    print(f"Effective baseline, Beff = {eff_baseline:g}")
    print("Centre frequencies:")
    for freq in freqs:
        print(f"f = {freq.to(u.MHz):g}  λ = {(c.c / freq).to(u.m):g}")
    width = (1 * u.rad * (c.c / freqs) / eff_baseline).decompose()
    print(f"... beam width ~ λ/Beff: {width.to(u.arcminute)}")

    # Define reference frame and time
    time = Time(args.time, format="isot", scale="utc")
    altaz_frame = AltAz(location=MWA_LOCATION, obstime=time)

    if args.plot:
        print("Plotting array layout...")
        plot_array_layout(
            context,
            extra_tile_flags=args.tile_flags,
            show_flagged_tiles=(not args.no_tile_flags),
        )
        plot_baseline_distribution(
            context,
            extra_tile_flags=args.tile_flags,
            show_flagged_tiles=(not args.no_tile_flags),
        )

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
    print(f"... took {t1 - t0} seconds")

    # In principle, allow the user to provide N inputs separated by spaces, or just
    # ask for M pointings around the source
    target_ras = []
    target_decs = []

    t0 = timer.time()
    print(
        "Creating sky position vectors from highest frequency and first look-direction..."
    )

    if args.use_wcs:
        print(
            "Generating a WCS grid around the central localisation position..."
        )
        print(
            f"   {look_positions[0].to_string('hmsdms', sep=':', precision=2)}"
        )
        print(f"Grid shape = {args.wcs_grid_size}")
        print(f"Pixel scale = {args.wcs_pixel_size} arcsec")

        grid_ra, grid_dec, wcs = generate_wcs_grid(
            look_positions[0],
            arcsec_per_pixel=args.wcs_pixel_size,
            image_size=args.wcs_grid_size,
        )
        target_positions = SkyCoord(
            grid_ra,
            grid_dec,
            frame="icrs",
            unit=("deg", "deg"),
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
    print(f"... took {t1 - t0} seconds")

    print("Converting to AltAz...")
    t0 = timer.time()
    target_positions_altaz = target_positions.transform_to(altaz_frame)
    t1 = timer.time()
    print(f"... took {t1 - t0} seconds")

    # Compute and store the primary beam map, if requested
    pbp_freq = []
    for j, freq in enumerate(freqs):
        if args.nopb:
            pbp_freq.append(None)
        else:
            # Compute the primary beam zenith-normalised power.
            print(f"Computing primary beam power at frequency = {freq}...")
            t0 = timer.time()
            pbp = get_primary_beam_power(
                context,
                freq.value,
                target_positions_altaz.alt.rad,
                target_positions_altaz.az.rad,
                stokes="I",
            )["I"].reshape(grid_ra.shape)
            pbp_freq.append(pbp)
            print(f"... primary beam max. in-field power = {pbp.max():.3f}")
            t1 = timer.time()
            print(f"... took {t1 - t0} seconds")
    pbp_freq = np.array(pbp_freq)

    if not args.nopb:
        print("Plotting primary beam map...")
        plot_primary_beam(
            context,
            pbp_freq[0, ...],
            grid_ra,
            grid_dec,
            [0.05, 0.1, 0.25, 0.5, 0.8, 1],
            wcs,
            target=look_positions[0],
        )

    # Start the loops over look-directions and compute the TABs
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
            print(f"Processing tied-array beam at frequency = {freq}")
            print("Computing array factors...")
            t0 = timer.time()
            # Compute the array factor (tied-array beam weighting factor).
            look_psi = calc_geometric_delays(
                tile_positions,
                freq.value,
                lp.alt.rad,
                lp.az.rad,
            )
            target_psi = calc_geometric_delays(
                tile_positions,
                freq.value,
                target_positions_altaz.alt.rad,
                target_positions_altaz.az.rad,
            )
            afp = calc_array_factor_power(look_psi, target_psi)
            t1 = timer.time()
            print(f"... took {t1 - t0} seconds")

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

    if args.plot:
        ctr_levels = [0.05, 0.1, 0.25, 0.5, 0.8, 1]
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
            wcs,
            scale_arcmin=round(width[0].to("arcmin").value, 1),
            label=tab_cbar_label,
            oname_suffix=oname_suffix,
        )

    tt1 = timer.time()
    print(f"Done!! (Took {tt1 - tt0} seconds.)\n")

    # Execute the localisation method using the TABs and detection data
    if args.localise:
        if args.detfile is not None:
            if args.truth is not None:
                true_coords = SkyCoord(
                    args.truth,
                    frame="icrs",
                    unit=("hourangle", "deg"),
                )
            else:
                true_coords = None
            loc, cov = localise(
                args.detfile,
                tabp_look,
                grid_ra,
                grid_dec,
                wcs,
                truth_coords=true_coords,
                window=regularisation_fn,
                zoom=args.zoom,
            )
            loc.savefig("localisation.png", dpi=200)
            cov.savefig("covariance.png", dpi=200)
        else:
            print("ERROR: No detection file provided.")


if __name__ == "__main__":
    main()
