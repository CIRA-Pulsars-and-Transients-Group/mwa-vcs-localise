#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import argparse
from .utils import TileArray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metafits")

    args = parser.parse_args()

    a = TileArray.from_metafits(args.metafits)
    print(a)


if __name__ == "__main__":
    main()
