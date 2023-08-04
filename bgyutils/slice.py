#!/usr/bin/env python3

from enum import Enum, unique
from typing import Optional
from numpy import ndarray
from matplotlib import pyplot

from structs import DataPacket


@unique
class SliceOrientation(Enum):
    XLine = 0
    Inline = 1
    Depth = 2


def slice_ndarray(array: ndarray, orientation: SliceOrientation, value: int) -> ndarray:
    """
    Slice a 3D ndarray along a given axis at a given value.
    """
    if orientation == SliceOrientation.XLine:
        return array[value, :, :]
    elif orientation == SliceOrientation.Inline:
        return array[:, value, :]
    elif orientation == SliceOrientation.Depth:
        return array[:, :, value]
    else:
        raise ValueError("Invalid orientation")


def slice_datapacket(
    datapack: DataPacket, orientation: SliceOrientation, value: int
) -> Optional[ndarray]:
    """
    Slice a DataPacket along a given axis at a given value.

    Returns None if the slice is out of bounds.
    """
    if (
        value < datapack.offset[orientation.value]
        or value
        >= datapack.offset[orientation.value] + datapack.size[orientation.value]
    ):
        return None
    value -= datapack.offset[orientation.value]
    return slice_ndarray(datapack.to_numpy(), orientation, value)

def show_slice(datapack: DataPacket, orientation: SliceOrientation, value: int) -> bool:
    """
    Slice a DataPacket and show the result using matplotlib.

    Returns False if the slice is out of bounds.
    """
    slice = slice_datapacket(datapack, orientation, value)
    if slice is None:
        return False

    if orientation == SliceOrientation.XLine:
        x_label = "Inline"
        y_label = "Depth"
        extent = [
            datapack.offset[SliceOrientation.Inline.value],
            datapack.offset[SliceOrientation.Inline.value] + datapack.size[SliceOrientation.Inline.value],
            datapack.offset[SliceOrientation.Depth.value],
            datapack.offset[SliceOrientation.Depth.value] + datapack.size[SliceOrientation.Depth.value],
        ]
    elif orientation == SliceOrientation.Inline:
        x_label = "Xline"
        y_label = "Depth"
        extent = [
            datapack.offset[SliceOrientation.XLine.value],
            datapack.offset[SliceOrientation.XLine.value] + datapack.size[SliceOrientation.XLine.value],
            datapack.offset[SliceOrientation.Depth.value],
            datapack.offset[SliceOrientation.Depth.value] + datapack.size[SliceOrientation.Depth.value],
        ]
    elif orientation == SliceOrientation.Depth:
        x_label = "Xline"
        y_label = "Inline"
        extent = [
            datapack.offset[SliceOrientation.XLine.value],
            datapack.offset[SliceOrientation.XLine.value] + datapack.size[SliceOrientation.XLine.value],
            datapack.offset[SliceOrientation.Inline.value],
            datapack.offset[SliceOrientation.Inline.value] + datapack.size[SliceOrientation.Inline.value],
        ]
    else:
        raise ValueError("Invalid orientation")

    pyplot.imshow(slice.T, extent=extent, cmap="gray", aspect="auto")
    pyplot.title(f"Slice at {orientation.name}={value}")
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.show()
    return True
