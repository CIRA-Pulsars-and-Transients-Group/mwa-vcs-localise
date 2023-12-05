#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import argparse
import mwalib
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
import matplotlib.pyplot as plt
from .utils import MWA_LOCATION
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
        help="Sky position to compute array factor given the look-direction "
        "(format: 'hh:mm:ss_dd:mm:ss').\nYou may provide multiple sky positions "
        "to sample, separating them by a single <space>.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the TAB power for each provided point.",
    )

    args = parser.parse_args()

    # Collect meta information and setup configuration.
    context = mwalib.MetafitsContext(args.metafits)
    time = Time(args.time, format="isot", scale="utc")
    altaz_frame = AltAz(location=MWA_LOCATION, obstime=time)

    look_ra, look_dec = args.look.split("_")
    # In principle, allow the user to provide N inputs separated by spaces
    target_ras = []
    target_decs = []
    for p in args.position.split(" "):
        target_ras.append(p.split("_")[0])
        target_decs.append(p.split("_")[1])

    look_position = SkyCoord(
        look_ra,
        look_dec,
        frame="icrs",
        unit=("hourangle", "deg"),
    )
    look_position_altaz = look_position.transform_to(altaz_frame)
    target_positions = SkyCoord(
        target_ras,
        target_decs,
        frame="icrs",
        unit=("hourangle", "deg"),
    )
    target_positions_altaz = target_positions.transform_to(altaz_frame)

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

    # Compute the primary beam zenith-normalised power.
    pbp = getPrimaryBeamPower(
        context,
        args.freq,
        target_positions_altaz.alt.rad,
        target_positions_altaz.az.rad,
    )

    # Finally, estimate the zenith-normalised tied-array beam power.
    tabp = afp * pbp

    if args.plot:
        plt.scatter(
            target_positions.ra.deg,
            target_positions.dec.deg,
            c=tabp,
            cmap=plt.get_cmap("Reds"),
            norm="linear",
        )
        plt.xlabel("RA (deg)")
        plt.ylabel("Dec (deg)")
        plt.colorbar(label="Zenith-normalised tied-array beam sensitivity")
        plt.show()


if __name__ == "__main__":
    main()
