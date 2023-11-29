#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import argparse
from .utils import Array


def main():
    parser = argparse.ArgumentParse()
    parser.add_argument("metafits")

    args = parser.parse_args()

    a = Array()
    print(a)


if __name__ == "__main__":
    main()
