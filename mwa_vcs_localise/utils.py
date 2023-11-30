#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

from attrs import define, field
import astropy.units as u
from astropy.io import fits
import numpy as np


@define
class TileArray:
    """A container class that defines a certain array configuration."""

    antenna_id: np.array  # int, antenna id numbers
    tile_names: np.chararray  # str, tile names
    tile_ids: np.array  # int, tile id numbers
    tile_eastings: np.array  # float, tile easting from array centre in m
    tile_northings: np.array  # float, tile northing from array centre in m
    tile_heights_asl: np.array  # float, tile height coordinate ABOVE SEA LEVEL in m
    tile_flags: np.array  # bool, whether the tiles are flagged (True) or not (False)
    centre_height_asl: float = 377.827  # array centre height ABOVE SEA LEVEL in m
    tile_heights: np.array = field(init=False)

    def __attrs_post_init__(self):
        self.tile_heights = self.tile_heights_asl - self.centre_height_asl

    @classmethod
    def from_metafits(cls, metafits: str):
        """A class method to initialise an instance of TileArray using
        information gathered from the provided metafits file.

        :param metafits: Path to the MWA observation metafits files
        :type metafits: str
        :return: An instance of the TileArray class
        :rtype: TileArray
        """
        with fits.open(metafits) as f:
            # Take every other entry because there are two polarisations
            ant_ids = f[1].data["Antenna"][::2]
            tile_ids = f[1].data["Tile"][::2]
            tile_names = f[1].data["TileName"][::2]
            e = f[1].data["East"][::2]
            n = f[1].data["North"][::2]
            h = f[1].data["Height"][::2]
            flags = f[1].data["Flag"][::2].astype(bool)

        return cls(ant_ids, tile_names, tile_ids, e, n, h, flags)
