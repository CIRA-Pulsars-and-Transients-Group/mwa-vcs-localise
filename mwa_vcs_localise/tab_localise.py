#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import argparse
import mwalib
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
from .utils import MWA_LOCATION
from .array_factor import calcGeometricDelays, calcArrayFactorPower
from .primary_beam import getPrimaryBeamPower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metafits")
    parser.add_argument(
        "--time", type=str, help="UTC time of observation (format: ISOT)."
    )
    parser.add_argument("--freq", type=float, help="Observing frequency in Hz.")
    parser.add_argument(
        "--look",
        "-l",
        type=str,
        help="Look-direction sky position (format: 'hh:mm:ss_dd:mm:ss')",
    )
    parser.add_argument(
        "--position",
        "-p",
        type=str,
        help="Sky position to compute array factor given the look-direction (format: 'hh:mm:ss_dd:mm:ss')",
    )

    args = parser.parse_args()

    # Collect meta information and setup configuration.
    context = mwalib.MetafitsContext(args.metafits)
    time = Time(args.time, format="isot", scale="utc")
    altaz_frame = AltAz(location=MWA_LOCATION, obstime=time)

    look_ra, look_dec = args.look.split("_")
    target_ra, target_dec = args.position.split("_")

    look_position, target_position = SkyCoord(
        [look_ra, target_ra],
        [look_dec, target_dec],
        frame="icrs",
        unit=("hourangle", "deg"),
    ).transform_to(altaz_frame)

    # Compute the array factor (tied-array beam weighting factor).
    look_psi = calcGeometricDelays(
        context,
        args.freq,
        look_position.alt.rad,
        look_position.az.rad,
    )
    target_psi = calcGeometricDelays(
        context,
        args.freq,
        target_position.alt.rad,
        target_position.az.rad,
    )

    afp = calcArrayFactorPower(look_psi, target_psi)
    print(afp)

    # Compute the primary beam zenith-normalised power.
    pbp = getPrimaryBeamPower(
        context,
        args.freq,
        target_position.alt.rad,
        target_position.az.rad,
    )
    print(pbp)

    # Finally, estimate the zenith-normalised tied-array beam power.
    tabp = afp * pbp
    print(tabp)


if __name__ == "__main__":
    main()
