#!/usr/bin/env python3

from os import PathLike
from typing import Union

from numpy import ndarray, flip
import segyio


def read_segy(
    file: Union[str, PathLike],
    ignore_geometry: bool = False,
    swap_inline_xline: bool = False,
) -> ndarray:
    """
    Read a SEG-Y file using segyio and return a 3d numpy array containing values of each point.
    """
    with segyio.open(file, ignore_geometry=ignore_geometry) as segyfile:
        data = segyio.tools.cube(segyfile)
        if swap_inline_xline:
            data = data.swapaxes(0, 1)
        # Flip the Z axis
        data = flip(data, axis=-1)
    return data
