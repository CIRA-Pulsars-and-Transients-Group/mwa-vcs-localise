#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import typing
from attrs import define, field
import astropy.units as u
from astropy.io import fits

# Define some type aliases for convenience
intvec: typing.TypeAlias = list[int]
fltvec: typing.TypeAlias = list[float]
strvec: typing.TypeAlias = list[str]


@define
class Array:
    array_config: str
    antenna_id: intvec
    tile_names: strvec
    tile_ids: list
    tile_eastings: fltvec  # tile easting from array centre
    tile_northings: fltvec  # tile northing from array centre
    tile_heights: fltvec  # tile height coordinate ABOVE SEA LEVEL
    array_centre_hasl: u.Quantity = 377.827 * u.m
