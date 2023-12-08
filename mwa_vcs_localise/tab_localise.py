#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
import time as timer
import argparse
import mwalib
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
        "-f", dest="freq", type=float, help="Observing frequency in Hz."
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
        "--plot",
        action="store_true",
        help="Whether to plot the TAB power for each provided point.",
    )

    args = parser.parse_args()

    # Collect meta information and setup configuration.
    context = mwalib.MetafitsContext(args.metafits)
    max_baseline, _, _ = find_max_baseline(context)
    fwhm = (1.22 * (sol.value / args.freq) / max_baseline) * u.rad
    print(f"maximum baseline (m): {max_baseline}")
    print(f"beam fwhm (arcmin): {fwhm.to(u.arcminute).value}")
    time = Time(args.time, format="isot", scale="utc")
    altaz_frame = AltAz(location=MWA_LOCATION, obstime=time)

    # Create the astrometric quantity for the beamformed target direction
    look_ra, look_dec = args.look.split("_")
    look_position = SkyCoord(
        look_ra,
        look_dec,
        frame="icrs",
        unit=("hourangle", "deg"),
    )
    look_position_altaz = look_position.transform_to(altaz_frame)

    # In principle, allow the user to provide N inputs separated by spaces, or just ask for M pointings around the source
    target_ras = []
    target_decs = []

    t0 = timer.time()
    print("Creating sky position samples...")
    if not args.position:
        target_positions = form_grid_positions(
            look_position,
            max_separation_arcsec=fwhm.to(u.arcsecond).value,
            nlayers=200,
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

    print("Computing array factors...")
    t0 = timer.time()
    # Compute the array factor (tied-array beam weighting factor).
    look_psi = calcGeometricDelays(
        context,
        args.freq,
        look_position_altaz.alt.rad,
        look_position_altaz.az.rad,
    )
    target_psi = calcGeometricDelays(
        context,
        args.freq,
        target_positions_altaz.alt.rad,
        target_positions_altaz.az.rad,
    )
    afp = calcArrayFactorPower(look_psi, target_psi)
    t1 = timer.time()
    print(f"... took {t1-t0} seconds")

    # Compute the primary beam zenith-normalised power.
    print("Computing primary beam power...")
    t0 = timer.time()
    pbp = getPrimaryBeamPower(
        context,
        args.freq,
        target_positions_altaz.alt.rad,
        target_positions_altaz.az.rad,
    )
    t1 = timer.time()
    print(f"... took {t1-t0} seconds")

    # Finally, estimate the zenith-normalised tied-array beam power.
    tabp = afp * pbp

    if args.plot:
        plt.scatter(
            target_positions.ra,
            target_positions.dec,
            c=tabp,
            cmap=plt.get_cmap("Reds"),
            norm="log",
            vmin=max(tabp.min(), 1e-3),
        )

        plt.xlabel("RA (deg)")
        plt.ylabel("Dec (deg)")
        plt.colorbar(label="Zenith-normalised tied-array beam sensitivity")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
