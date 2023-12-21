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
from .utils import MWA_LOCATION, form_grid_positions, find_max_baseline
from .array_factor import calcGeometricDelays, calcArrayFactorPower
from .primary_beam import getPrimaryBeamPower


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
        If no argument provided, will sample the sky around position based on
        an estimate of the FWHM.""",
        default=None,
    )
    parser.add_argument(
        "--nlayers",
        type=int,
        help="""If not argument provided to -P, this option sets how many
        circular layers of FWHM-sized cells around the central position to
        calculate.""",
        default=100,
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

    # Collect meta information and setup configuration.
    context = mwalib.MetafitsContext(args.metafits)
    max_baseline, _, _ = find_max_baseline(context)
    fwhm = (1.22 * (sol.value / freqs) / max_baseline) * u.rad
    print(f"maximum baseline (m): {max_baseline}")
    print(f"beam fwhm (arcmin): {fwhm.to(u.arcminute).value}")
    time = Time(args.time, format="isot", scale="utc")
    altaz_frame = AltAz(location=MWA_LOCATION, obstime=time)

    # Create the astrometric quantity for the beamformed target direction
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
    look_positions_altaz = look_positions.transform_to(altaz_frame)

    # In principle, allow the user to provide N inputs separated by spaces, or just
    # ask for M pointings around the source
    target_ras = []
    target_decs = []

    target_positions_freq = []

    t0 = timer.time()
    print(
        "Creating sky position samples from highest frequency and first look-direction..."
    )
    if not args.position:
        target_positions = form_grid_positions(
            look_positions[0],
            max_separation_arcsec=(min(fwhm)).to(u.arcsecond).value,
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
    for i, lp in enumerate(look_positions_altaz):
        print(f"\nProcessing look-direction = {look_positions[i]}")
        tabp_freq = []
        for j, freq in enumerate(freqs):
            print(f"Processing frequency = {freq} Hz\n")
            print("Computing array factors...")
            t0 = timer.time()
            # Compute the array factor (tied-array beam weighting factor).
            look_psi = calcGeometricDelays(
                context,
                freq,
                lp.alt.rad,
                lp.az.rad,
            )
            target_psi = calcGeometricDelays(
                context,
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
                print(f"... primary beam max. power = {pbp.max()}")
                t1 = timer.time()
                print(f"... took {t1-t0} seconds")
            else:
                pbp = 1.0

            # Finally, estimate the zenith-normalised tied-array beam power.
            tabp = afp * pbp
            tabp_freq.append(tabp)
        tabp_look.append(tabp_freq)

    tabp_look = np.array(tabp_look)
    print(tabp_look.shape)

    if args.plot:
        if args.nopb:
            label = "Array factor power"
        else:
            label = "Zenith-normalised tied-array beam sensitivity"

        print("Plotting sky map...")
        product = np.prod(tabp_look.mean(axis=1), axis=0)
        tab_map = plt.scatter(
            target_positions.ra.deg,
            target_positions.dec.deg,
            c=product,
            s=50,
            marker="o",
            alpha=0.5,
            cmap="viridis",
            norm="log",
            # vmin=min(product.min(), 1e-3),
        )
        plt.scatter(
            look_positions.ra.deg,
            look_positions.dec.deg,
            c="magenta",
            marker="x",
        )

        plt.xlabel("RA (deg)")
        plt.ylabel("Dec (deg)")
        cbar = plt.colorbar(tab_map, label=label)
        plt.tight_layout()
        t1 = timer.time()
        print(f"... took {t1-t0} seconds")

        plt.show()


if __name__ == "__main__":
    main()
