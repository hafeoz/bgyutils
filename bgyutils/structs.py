#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, unique
from struct import pack, unpack
from sys import byteorder
from typing import BinaryIO, Dict, List, Tuple
from numpy import frombuffer, ndarray, single
from humanize import naturalsize


@unique
class DataPacketEncodingFormat(Enum):
    """
    Data format for a BGY data packet.
    """

    FLOAT_PLAIN_LE = 0  # 32-bit float, little-endian, no compression

    def encode_data(self, data: ndarray, use_legacy_encoder: bool = False) -> bytes:
        """
        Encode the data in the given format.
        """

        if self == DataPacketEncodingFormat.FLOAT_PLAIN_LE:
            data = data.astype(single)
            if use_legacy_encoder:
                serialized_array = bytearray()
                for xline in range(data.shape[0]):
                    for iline in range(data.shape[1]):
                        for depth in range(data.shape[2]):
                            data_point = data[xline, iline, depth]
                            if byteorder == "big":
                                data_point = data_point.byteswap()
                            serialized_array.extend(data_point.tobytes())
                serialized = bytes(serialized_array)
            else:
                data = data.flatten()
                if byteorder == "big":
                    data = data.byteswap()
                serialized = data.tobytes()
        else:
            raise NotImplementedError(f"Unsupported data format: {self}")

        return serialized

    def decode_data(self, data: bytes, shape: Tuple[int, int, int]) -> ndarray:
        """
        Decode the data in the given format.
        """
        if self == DataPacketEncodingFormat.FLOAT_PLAIN_LE:
            deserialized = frombuffer(data, dtype=single)
            assert deserialized.shape[0] == shape[0] * shape[1] * shape[2]
            if byteorder == "big":
                deserialized = deserialized.byteswap()
            deserialized = deserialized.reshape(shape)
        else:
            raise NotImplementedError(f"Unsupported data format: {self}")

        return deserialized

    def to_bytes(self) -> bytes:
        """
        Convert the data format to bytes.
        """
        return self.value.to_bytes(1, byteorder="little")

    @classmethod
    def from_io(cls, data: BinaryIO) -> DataPacketEncodingFormat:
        """
        Read the data format from a binary IO.
        """
        return cls(int.from_bytes(data.read(1), byteorder="little"))


@dataclass
class DataPacket:
    """
    A BGY data packet.

    It contains both the raw data and the metadata associated.
    """

    # Metadata
    offset: Tuple[
        int, int, int
    ]  # Crossline, inline and depth offset of the data packet in the whole volume
    size: Tuple[
        int, int, int
    ]  # Crossline, inline and depth size of the data packet in the whole volume
    data_shape: Tuple[
        int, int, int
    ]  # Crossline, inline and depth size of the data packet. This may be different from the size if the data packet is downsampled.
    encoding_format: DataPacketEncodingFormat  # Data format of the data packet

    # Data
    data: bytes  # Raw data of the data packet

    def to_numpy(self) -> ndarray:
        """
        Convert the data packet to a numpy array.
        """
        return self.encoding_format.decode_data(self.data, self.data_shape)

    def to_bytes(self) -> bytes:
        """
        Convert the data packet to bytes.
        """
        serialized = bytearray()
        serialized.extend(self.offset[0].to_bytes(4, byteorder="little"))
        serialized.extend(self.offset[1].to_bytes(4, byteorder="little"))
        serialized.extend(self.offset[2].to_bytes(4, byteorder="little"))
        serialized.extend(self.size[0].to_bytes(4, byteorder="little"))
        serialized.extend(self.size[1].to_bytes(4, byteorder="little"))
        serialized.extend(self.size[2].to_bytes(4, byteorder="little"))
        serialized.extend(self.data_shape[0].to_bytes(4, byteorder="little"))
        serialized.extend(self.data_shape[1].to_bytes(4, byteorder="little"))
        serialized.extend(self.data_shape[2].to_bytes(4, byteorder="little"))
        serialized.extend(self.encoding_format.to_bytes())
        serialized.extend(len(self.data).to_bytes(4, byteorder="little"))
        serialized.extend(self.data)
        return bytes(serialized)

    @classmethod
    def from_io(cls, data: BinaryIO) -> DataPacket:
        """
        Read the data packet from a binary IO.
        """
        offset = (
            int.from_bytes(data.read(4), byteorder="little"),
            int.from_bytes(data.read(4), byteorder="little"),
            int.from_bytes(data.read(4), byteorder="little"),
        )
        size = (
            int.from_bytes(data.read(4), byteorder="little"),
            int.from_bytes(data.read(4), byteorder="little"),
            int.from_bytes(data.read(4), byteorder="little"),
        )
        data_shape = (
            int.from_bytes(data.read(4), byteorder="little"),
            int.from_bytes(data.read(4), byteorder="little"),
            int.from_bytes(data.read(4), byteorder="little"),
        )
        encoding_format = DataPacketEncodingFormat.from_io(data)
        data_size = int.from_bytes(data.read(4), byteorder="little")
        raw_data = data.read(data_size)
        return cls(
            offset=offset,
            size=size,
            data_shape=data_shape,
            encoding_format=encoding_format,
            data=raw_data,
        )

    def description(self) -> str:
        """
        Return a human-readable description of the data packet.
        """
        info = "Offset: {}x{}x{}\n".format(*self.offset)
        info += "Size: {}x{}x{}\n".format(*self.size)
        info += "Data shape: {}x{}x{}\n".format(*self.data_shape)
        info += "Encoding format: {}\n".format(self.encoding_format)
        info += "Data size: {} bytes\n".format(naturalsize(len(self.data)))
        return info


@dataclass
class Bgy:
    """
    A BGY file.

    It contains some metadata, and a series of data packets.
    """

    # Metadata
    volume_size: Tuple[int, int, int]  # Crossline, inline and depth size of the volume
    resolution: Tuple[
        float, float, float
    ]  # Gap between each data point in crossline, inline and depth direction (in meters)
    value_range: Tuple[float, float]  # Minimum and maximum value of the data.
    metadata: Dict[str, str]  # Additional metadata

    # Data
    data_packets: List[DataPacket]  # List of data packets

    def to_bytes(self) -> bytes:
        """
        Convert the BGY file to bytes.
        """
        serialized = bytearray()
        serialized.extend(b"BGY")
        serialized.extend(self.volume_size[0].to_bytes(4, byteorder="little"))
        serialized.extend(self.volume_size[1].to_bytes(4, byteorder="little"))
        serialized.extend(self.volume_size[2].to_bytes(4, byteorder="little"))
        serialized.extend(pack("<f", self.resolution[0]))
        serialized.extend(pack("<f", self.resolution[1]))
        serialized.extend(pack("<f", self.resolution[2]))
        serialized.extend(pack("<f", self.value_range[0]))
        serialized.extend(pack("<f", self.value_range[1]))
        serialized.extend(len(self.metadata).to_bytes(4, byteorder="little"))
        for key, value in self.metadata.items():
            serialized.extend(len(key).to_bytes(4, byteorder="little"))
            serialized.extend(key.encode("utf-8"))
            serialized.extend(len(value).to_bytes(4, byteorder="little"))
            serialized.extend(value.encode("utf-8"))
        serialized.extend(len(self.data_packets).to_bytes(4, byteorder="little"))
        for data_packet in self.data_packets:
            serialized.extend(data_packet.to_bytes())
        return bytes(serialized)

    @classmethod
    def from_io(cls, data: BinaryIO) -> Bgy:
        """
        Read the BGY file from a binary IO.
        """
        magic = data.read(3)
        if magic != b"BGY":
            raise ValueError("Invalid BGY file")
        volume_size = (
            int.from_bytes(data.read(4), byteorder="little"),
            int.from_bytes(data.read(4), byteorder="little"),
            int.from_bytes(data.read(4), byteorder="little"),
        )
        resolution = (
            unpack("<f", data.read(4))[0],
            unpack("<f", data.read(4))[0],
            unpack("<f", data.read(4))[0],
        )
        value_range = (
            unpack("<f", data.read(4))[0],
            unpack("<f", data.read(4))[0],
        )
        metadata = {}
        metadata_size = int.from_bytes(data.read(4), byteorder="little")
        for _ in range(metadata_size):
            key_size = int.from_bytes(data.read(4), byteorder="little")
            key = data.read(key_size).decode("utf-8")
            value_size = int.from_bytes(data.read(4), byteorder="little")
            value = data.read(value_size).decode("utf-8")
            metadata[key] = value
        data_packet_size = int.from_bytes(data.read(4), byteorder="little")
        data_packets = []
        for _ in range(data_packet_size):
            data_packets.append(DataPacket.from_io(data))
        return cls(
            volume_size=volume_size,
            resolution=resolution,
            value_range=value_range,
            metadata=metadata,
            data_packets=data_packets,
        )

    def description(self) -> str:
        """
        Return a description of the BGY file.
        """
        info = "Size: {}x{}x{}\n".format(*self.volume_size)
        info += "Resolution: {}x{}x{}\n".format(*self.resolution)
        info += "Value range: {} to {}\n".format(*self.value_range)
        info += "Metadata:\n"
        for key, value in self.metadata.items():
            info += "  {}: {}\n".format(key, value)
        info += "Data packets: {}\n".format(len(self.data_packets))
        for data_packet in self.data_packets:
            info += "  {}\n".format(data_packet.description().replace("\n", "\n  "))
        return info
