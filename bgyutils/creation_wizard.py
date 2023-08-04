#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, unique
import math
from os import PathLike
from typing import Callable, List, Optional, Tuple, TypeVar, Union
from pathlib import Path
from matplotlib.ticker import PercentFormatter
from numpy import ndarray
import numpy
from yaspin import yaspin
from matplotlib import pyplot
from humanize import naturalsize

from read_segy import read_segy
from structs import DataPacketEncodingFormat, Bgy, DataPacket

T = TypeVar("T")


def ask_user(
    prompt: str,
    parser: Callable[[str], T] = str,
    description: Optional[str] = None,
    default: Optional[T] = None,
    choices: Optional[List[str]] = None,
) -> T:
    """
    Ask the user a question, and return the answer.
    """
    if description:
        print(description)
    while True:
        try:
            answer = input(
                prompt
                + (" (choices: " + ", ".join(choices) + ")" if choices else "")
                + (f" (default: {default})" if default is not None else "")
                + ": "
            )
            if not answer.strip() and default is not None:
                return default
            return parser(answer)
        except ValueError as e:
            print(e)


def percentiles(values: ndarray, percentiles: List[float]) -> List[Tuple[float, float]]:
    """
    Calculate percentiles of a list of values.
    """
    with yaspin(text="Calculating percentiles...", timer=True) as spinner:
        spinner.text = "Flattening values..."
        values = values.flatten()
        calculated_percentiles = []
        for percentile in percentiles:
            spinner.text = f"Calculating percentile {percentile}..."
            calculated_percentiles.append(
                (percentile, float(numpy.percentile(values, percentile)))
            )
        spinner.color = "green"
        spinner.ok("✔")
    return calculated_percentiles


def path_parser(path_str: str) -> Path:
    """
    Parse a path.

    Raises an exception if the path does not exist.
    """
    if not path_str.strip():
        raise ValueError("Path cannot be empty.")
    path = Path(path_str)
    if not path.is_file():
        raise ValueError(f"Path {path_str} does not exist or is not a file.")
    return path


def bool_parser(bool_str: str) -> bool:
    """
    Parse a boolean.
    """
    if bool_str.lower() in ["true", "yes", "y", "true", "t"]:
        return True
    if bool_str.lower() in ["false", "no", "n", "false", "f"]:
        return False
    raise ValueError(
        f"Invalid boolean value {bool_str}. Possible values: true/yes/y/true/t/false/no/n/false/f"
    )


@dataclass
class DataPacketCreationParameters:
    """
    Parameters required for creating a data packet.
    """

    offset: Tuple[int, int, int]
    size: Tuple[int, int, int]
    data_shape: Tuple[int, int, int]
    data_type: DataPacketEncodingFormat

    def description(self) -> str:
        """
        Get a description of the parameters.
        """
        return f"offset: {self.offset}, size: {self.size}, data_shape: {self.data_shape}, data_type: {self.data_type}, estimated size: {naturalsize(self.data_shape[0] * self.data_shape[1] * self.data_shape[2] * 4)}"

    def encode_data_packet(self, data: ndarray) -> DataPacket:
        """
        Encode a data packet using the parameters.
        """
        # Convert data to the correct type
        if self.data_type == DataPacketEncodingFormat.FLOAT_PLAIN_LE:
            data = data.astype(numpy.float32)
        # Slice the data
        data = data[
            self.offset[0] : self.offset[0] + self.size[0],
            self.offset[1] : self.offset[1] + self.size[1],
            self.offset[2] : self.offset[2] + self.size[2],
        ]
        # If data_shape is not the same as size, use nearest neighbor interpolation to resize the data
        if self.data_shape != self.size:
            new_data = numpy.zeros(self.data_shape, dtype=data.dtype)
            # TODO: Use a better algorithm. Maybe https://stackoverflow.com/a/73662154 ?
            for x in range(self.data_shape[0]):
                for y in range(self.data_shape[1]):
                    for z in range(self.data_shape[2]):
                        new_data[x, y, z] = data[
                            x * self.size[0] // self.data_shape[0],
                            y * self.size[1] // self.data_shape[1],
                            z * self.size[2] // self.data_shape[2],
                        ]
            data = new_data

        return DataPacket(
            offset=self.offset,
            size=self.size,
            data_shape=self.data_shape,
            encoding_format=self.data_type,
            data=self.data_type.encode_data(data),
        )


@unique
class DataPacketCreationStrategy(Enum):
    SMART = 0
    SIMPLE = 1
    MANUAL = 2
    SIMPLE_THUMBNAIL = 3

    @classmethod
    def from_str(cls, value: str) -> DataPacketCreationStrategy:
        """
        Get a DataPacketCreationStrategy from a string.
        """
        if value.lower() == "smart":
            return cls.SMART
        elif value.lower() == "simple":
            return cls.SIMPLE
        elif value.lower() == "manual":
            return cls.MANUAL
        elif value.lower() == "simple_thumbnail":
            return cls.SIMPLE_THUMBNAIL
        else:
            raise ValueError(
                f"Invalid DataPacketCreationStrategy {value}. Possible values: smart/simple/manual/simple_thumbnail"
            )

    @classmethod
    def choices(cls) -> List[str]:
        """
        Get a list of possible choices.
        """
        return ["smart", "simple", "manual", "simple_thumbnail"]

    @staticmethod
    def smart_data_packet_creation(
        volume_shape: Tuple[int, int, int],
        thumbnail_bytes: int,
        packet_bytes: int,
    ) -> List[DataPacketCreationParameters]:
        """
        Creating data packets in a smart way.
        """
        packets = []

        # Create an thumbnail packet that covers the entire volume
        # Scale volume_shape to make it fit into thumbnail_size
        downsampling_factor = math.pow(
            thumbnail_bytes / (volume_shape[0] * volume_shape[1] * volume_shape[2]),
            1 / 3,
        )
        thumbnail_shape = (
            math.ceil(volume_shape[0] * downsampling_factor),
            math.ceil(volume_shape[1] * downsampling_factor),
            math.ceil(volume_shape[2] * downsampling_factor),
        )
        packets.append(
            DataPacketCreationParameters(
                offset=(0, 0, 0),
                size=volume_shape,
                data_shape=thumbnail_shape,
                data_type=DataPacketEncodingFormat.FLOAT_PLAIN_LE,
            )
        )

        cube_shape = volume_shape
        # While the downsampling factor is greater than 1, split the volume into 8 parts, and create packets for each part
        with yaspin(
            text="Splitting volume into packets...", timer=True
        ).shark as spinner:
            while downsampling_factor < 1:
                downsampling_factor = math.pow(
                    packet_bytes / (cube_shape[0] * cube_shape[1] * cube_shape[2]),
                    1 / 3,
                )
                spinner.text = f"Splitting volume into packets... (downsampling factor: {downsampling_factor:.2f}; cube shape: {cube_shape})"
                for x in range(0, volume_shape[0], cube_shape[0]):
                    for y in range(0, volume_shape[1], cube_shape[1]):
                        for z in range(0, volume_shape[2], cube_shape[2]):
                            size = (
                                min(cube_shape[0], volume_shape[0] - x),
                                min(cube_shape[1], volume_shape[1] - y),
                                min(cube_shape[2], volume_shape[2] - z),
                            )
                            packets.append(
                                DataPacketCreationParameters(
                                    offset=(x, y, z),
                                    size=size,
                                    data_shape=(
                                        math.ceil(size[0] * downsampling_factor),
                                        math.ceil(size[1] * downsampling_factor),
                                        math.ceil(size[2] * downsampling_factor),
                                    ),
                                    data_type=DataPacketEncodingFormat.FLOAT_PLAIN_LE,
                                )
                            )
                cube_shape = (
                    cube_shape[0] // 2,
                    cube_shape[1] // 2,
                    cube_shape[2] // 2,
                )

        return packets

    @staticmethod
    def simple_data_packet_creation(
        volume_shape: Tuple[int, int, int]
    ) -> List[DataPacketCreationParameters]:
        """
        Create one data packet that covers the entire volume.
        """
        return [
            DataPacketCreationParameters(
                offset=(0, 0, 0),
                size=volume_shape,
                data_shape=volume_shape,
                data_type=DataPacketEncodingFormat.FLOAT_PLAIN_LE,
            )
        ]

    @staticmethod
    def simple_thumbnail_data_packet_creation(
        volume_shape: Tuple[int, int, int],
        thumbnail_bytes: int,
    ) -> List[DataPacketCreationParameters]:
        """
        Create two data packets: one thumbnail and one regular packet.
        """
        downsampling_factor = math.pow(
            thumbnail_bytes / (volume_shape[0] * volume_shape[1] * volume_shape[2]),
            1 / 3,
        )
        thumbnail_shape = (
            math.ceil(volume_shape[0] * downsampling_factor),
            math.ceil(volume_shape[1] * downsampling_factor),
            math.ceil(volume_shape[2] * downsampling_factor),
        )
        return [
            DataPacketCreationParameters(
                offset=(0, 0, 0),
                size=volume_shape,
                data_shape=thumbnail_shape,
                data_type=DataPacketEncodingFormat.FLOAT_PLAIN_LE,
            ),
            DataPacketCreationParameters(
                offset=(0, 0, 0),
                size=volume_shape,
                data_shape=volume_shape,
                data_type=DataPacketEncodingFormat.FLOAT_PLAIN_LE,
            ),
        ]

    @staticmethod
    def manual_data_packet_creation(
        volume_shape: Tuple[int, int, int]
    ) -> List[DataPacketCreationParameters]:
        """
        Create data packets in a manual way.
        """
        packets = []
        while True:
            offset = [
                ask_user(f"Enter {axis} offset", int, default=0)
                for axis in ["X", "Y", "Z"]
            ]
            size = [
                ask_user(f"Enter {axis} size", int, default=volume_shape[i])
                for i, axis in enumerate(["X", "Y", "Z"])
            ]
            data_shape = [
                ask_user(f"Enter {axis} data shape", int, default=volume_shape[i])
                for i, axis in enumerate(["X", "Y", "Z"])
            ]
            packets.append(
                DataPacketCreationParameters(
                    offset=tuple(offset),
                    size=tuple(size),
                    data_shape=tuple(data_shape),
                    data_type=DataPacketEncodingFormat.FLOAT_PLAIN_LE,
                )
            )
            if not ask_user("Create another data packet?", bool_parser, default=False):
                break
        return packets

    def create_data_packets(
        self, volume_shape: Tuple[int, int, int]
    ) -> List[DataPacketCreationParameters]:
        """
        Create data packets.
        """
        if self == DataPacketCreationStrategy.SMART:
            while True:
                thumbnail_bytes = ask_user(
                    "Enter number of values in thumbnail", int, default=128**3
                )
                print(f"Estimated thumbnail size: {naturalsize(thumbnail_bytes * 4)}")
                confirm = ask_user("Is this correct?", bool_parser, default=True)
                if confirm:
                    break
            while True:
                packet_bytes = ask_user(
                    "Enter number of values in data packet", int, default=1024**3
                )
                print(f"Estimated data packet size: {naturalsize(packet_bytes * 4)}")
                confirm = ask_user("Is this correct?", bool_parser, default=True)
                if confirm:
                    break
            return self.smart_data_packet_creation(
                volume_shape, thumbnail_bytes, packet_bytes
            )
        elif self == DataPacketCreationStrategy.SIMPLE:
            return self.simple_data_packet_creation(volume_shape)
        elif self == DataPacketCreationStrategy.MANUAL:
            return self.manual_data_packet_creation(volume_shape)
        elif self == DataPacketCreationStrategy.SIMPLE_THUMBNAIL:
            while True:
                thumbnail_bytes = ask_user(
                    "Enter number of values in thumbnail", int, default=128**3
                )
                print(f"Estimated thumbnail size: {naturalsize(thumbnail_bytes * 4)}")
                confirm = ask_user("Is this correct?", bool_parser, default=True)
                if confirm:
                    break
            return self.simple_thumbnail_data_packet_creation(
                volume_shape, thumbnail_bytes
            )
        else:
            raise ValueError(f"Invalid DataPacketCreationStrategy {self}")


def creation_wizard(bgy_file_name: Union[str, PathLike]):
    """
    Wizard to create a new bgy file.
    """
    print("Welcome to the BGY creation wizard!")
    print("This wizard will help you create a new BGY file.")
    print("You can always press Ctrl+C to exit the wizard.")
    print("")

    print("Stage 1/2: Metadata")
    segy_file_name = ask_user(
        "Enter the name of the SEG-Y file to convert", path_parser
    )
    segy_file_ignore_geometry = ask_user(
        "Ignore geometry in SEG-Y file?",
        bool_parser,
        default=False,
        description="Ignoring geometry will load the SEG-Y faster and more fault-tolearant, but some structures may be lost or parsed incorrectly.",
    )
    segy_file_swap_inline_xline = ask_user(
        "Swap inline and crossline axes?",
        bool_parser,
        default=False,
        description="SEG-Y is a pretty loose standard, and some files may have the inline and crossline axes swapped. If you see that the geometry is incorrect, try swapping the axes.",
    )
    with yaspin(text="Loading SEG-Y file...", timer=True) as spinner:
        segy_file = read_segy(
            segy_file_name,
            ignore_geometry=segy_file_ignore_geometry,
            swap_inline_xline=segy_file_swap_inline_xline,
        )
        spinner.color = "green"
        spinner.ok("✔")

    resolution = tuple(
        [
            ask_user(f"Enter the {axis} resolution (in meters)", float, default=1.0)
            for axis in ["Crossline", "Inline", "Depth"]
        ]
    )

    metadata = {}
    while True:
        metadata_key = ask_user(
            "Enter metadata key (leave empty to stop entering metadata)", str
        )
        if metadata_key == "":
            break
        metadata_value = ask_user("Enter metadata value", str)
        metadata[metadata_key] = metadata_value

    data_percentiles_calculate = ask_user(
        "Calculate value percentiles?",
        bool_parser,
        default=True,
        description="Maximum and minimum values needs to be specified. Calculating the percentiles may be useful to determine these values, but it may take a long time.",
    )
    if data_percentiles_calculate:
        data_percentiles = percentiles(segy_file, [0, 1, 5, 10, 90, 95, 99, 100])
        for percentile in data_percentiles:
            print(f"Data percentile {percentile[0]}: {percentile[1]}")
        data_percentiles_show_plot = ask_user(
            "Show data percentiles plot?", bool_parser, default=True
        )
        if data_percentiles_show_plot:
            data_percentiles_len = segy_file.size
            pyplot.hist(
                segy_file.flatten(),
                bins=100,
                weights=numpy.ones(data_percentiles_len) / data_percentiles_len,
            )
            for percentile in data_percentiles:
                pyplot.axvline((int)(percentile[1]), color="red")
            pyplot.gca().yaxis.set_major_formatter(PercentFormatter(1))
            pyplot.show()
        min_max = (data_percentiles[0][1], data_percentiles[-1][1])
    else:
        min_max = (0, 100)
    value_range = tuple(
        [
            ask_user(
                f"Enter the {r} value of the data",
                float,
                default=i,
            )
            for (r, i) in zip(["Min", "Max"], min_max)
        ]
    )

    print("Stage 2/2: Creating data packets")
    while True:
        data_packet_creation_strategy = ask_user(
            "Enter data packet creation strategy",
            DataPacketCreationStrategy.from_str,
            "Data in BGY files is stored in data packets to allow for efficient loading. There are three methods to create data packets:\nSimple: create one data packet that covers the entire volume.\nSmart: create data packets of a fixed size, starting with a thumbnail and then increasing the size until the entire volume is covered.\nManual: create data packets manually by specifying the offset, size and data shape of each data packet.\nSimpleThumbnail: create one data packet that covers the entire volume, but also create a thumbnail data packet.",
            default=DataPacketCreationStrategy.SIMPLE,
            choices=DataPacketCreationStrategy.choices(),
        )
        data_packets = data_packet_creation_strategy.create_data_packets(
            segy_file.shape
        )
        print(f"Created {len(data_packets)} data packets.")
        for i, data_packet in enumerate(data_packets):
            print(f"Data packet {i+1}: {data_packet.description()}")
        data_packet_creation_confirm = ask_user(
            "Confirm data packet creation?", bool_parser, default=True
        )
        if data_packet_creation_confirm:
            break

    with yaspin(text="Creating BGY file...", timer=True) as spinner:
        encoded_data_packets = []
        for i, data_packet in enumerate(data_packets):
            spinner.text = f"Encoding data packet {i+1}/{len(data_packets)}..."
            encoded_data_packets.append(data_packet.encode_data_packet(segy_file))

        spinner.text = "Creating BGY..."
        bgy_file = Bgy(
            volume_size=segy_file.shape,
            resolution=resolution,
            value_range=value_range,
            metadata=metadata,
            data_packets=encoded_data_packets,
        )
        bgy_file_bytes = bgy_file.to_bytes()

        spinner.text = "Writing BGY file..."
        with open(bgy_file_name, "wb") as bgy_file_handle:
            bgy_file_handle.write(bgy_file_bytes)

        spinner.color = "green"
        spinner.ok("✔")
