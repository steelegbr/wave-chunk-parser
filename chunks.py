from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from exceptions import ExportExtendedFormatException, InvalidHeaderException
import numpy as np
from struct import unpack, pack
from typing import BinaryIO, List, Tuple
from utils import seek_and_read


class Chunk(ABC):
    """
    Base class for wave file chunks. We base all other chunks on this.
    """

    OFFSET_CHUNK_CONTENT = 8

    @classmethod
    def read_header(cls, file_handle: BinaryIO, offset: int) -> Tuple[str, int]:
        """
        Reads the headed from a chunk.

        Args:
            file_handle (BinaryIO): The IO stream to read from.
            offset (int): The offset to read at.

        Returns:
            Tuple[str, int]: The name of the chunk and the declared length.
        """

        return unpack(
            "<4sI", seek_and_read(file_handle, offset, cls.OFFSET_CHUNK_CONTENT)
        )

    @property
    @abstractmethod
    def get_name(self) -> str:  # pragma: no cover
        """
        Returns the name of the chunk type.
        """

        raise NotImplementedError

    @abstractmethod
    def to_bytes(self) -> List[bytes]:  # pragma: no cover
        """
        Encodes the chunck to a byte array for writing to a file.

        Returns:
            List[bytes]: The encoded chunk.
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_file(cls, file_handle: BinaryIO, offset: int) -> Chunk:  # pragma: no cover
        """
        Creates a chunk from a file.

        Args:
            file_handle (BinaryIO): The file handle to use.
            offset (int): The offset in the file this chunk is at.

        Returns:
            Chunk: We expect a subclass of Chunk to be returned.
        """
        raise NotImplementedError


class WaveFormat(Enum):
    """
    Supported wave formats.
    """

    PCM = 1
    FLOAT = 3
    A_LAW = 6
    MU_LAW = 7
    EXTENDED = 65534


class FormatChunk(Chunk):
    """
    The format chunk defines how the audio is encoded in a wave file.
    """

    __format: WaveFormat
    __extended: bool
    __channels: int
    __sample_rate: int
    __bits_per_sample: int

    LENGTH_CHUNK = 24
    LENGTH_STANDARD_SIZE = 16
    HEADER_FORMAT = b"fmt "

    def __init__(
        self,
        wave_format: WaveFormat,
        extended: bool,
        channels: int,
        sample_rate: int,
        bits_per_sample: int,
    ):
        """
        Creates a new instance of the format block.

        Args:
            format (WaveFormat): The format the audio is encoded in.
            extended (bool): Indicated if we have an extended audio file.
            channels (int): The number of channels in the file.
            sample_rate (int): The sample rate.
            bits_per_sample (int): The number of bits in each sample.
        """

        self.__format = wave_format
        self.__extended = extended
        self.__channels = channels
        self.__sample_rate = sample_rate
        self.__bits_per_sample = bits_per_sample

    @classmethod
    def from_file(cls, file_handle: BinaryIO, offset: int) -> FormatChunk:

        # Sanity check

        (header_str, length) = cls.read_header(file_handle, offset)

        if not header_str == cls.HEADER_FORMAT:
            raise InvalidHeaderException("Format chunk must start with fmt")

        # Check the length

        extended = length > cls.LENGTH_STANDARD_SIZE

        # Read from the chunk

        (handle, channels, sample_rate, _, _, bits_per_sample,) = unpack(
            "<HHIIHH",
            seek_and_read(
                file_handle,
                offset + cls.OFFSET_CHUNK_CONTENT,
                cls.LENGTH_CHUNK - cls.OFFSET_CHUNK_CONTENT,
            ),
        )

        # Read the format

        wave_format = WaveFormat(handle)

        # Generate our object

        return FormatChunk(
            wave_format, extended, channels, sample_rate, bits_per_sample
        )

    @property
    def format(self) -> WaveFormat:
        """
        Indicates the format the audio is encoded in.
        """
        return self.__format

    @property
    def channels(self) -> int:
        """
        The number of audio channels.
        """
        return self.__channels

    @property
    def sample_rate(self) -> int:
        """
        The sample rate.
        """
        return self.__sample_rate

    @property
    def bits_per_sample(self) -> int:
        """
        The number of bits used for each sample.
        """
        return self.__bits_per_sample

    @property
    def byte_rate(self) -> int:
        """
        The bytes per second this file is encoded at.
        """
        return self.sample_rate * self.channels * self.bits_per_sample // 8

    @property
    def block_align(self) -> int:
        """
        The block alignment for the file.
        """
        return self.channels * self.bits_per_sample // 8

    @property
    def extended(self) -> bool:
        """
        Indicates if the header is extended or not.
        """
        return self.__extended

    @property
    def get_name(self) -> str:
        return self.HEADER_FORMAT

    def to_bytes(self) -> List[bytes]:

        # Sanity check

        if self.extended:
            raise ExportExtendedFormatException(
                "We don't support converting extended format headers to binary blobs."
            )

        # Build up our chunk

        return pack(
            "<4sIHHIIHH",
            self.HEADER_FORMAT,
            self.LENGTH_STANDARD_SIZE,
            self.format.value,
            self.channels,
            self.sample_rate,
            self.byte_rate,
            self.block_align,
            self.bits_per_sample,
        )


class DataChunk(Chunk):
    """
    The data chunk holding the actual audio sample vectors.
    """

    __samples: np.ndarray[np.int]

    HEADER_DATA = b"data"

    def __init__(self, samples: np.ndarray[np.int]):
        """
        Creates a new instance of the data chunk.

        Args:
            samples (np.ndarray[np.int]): The samples to work with.
        """

        self.__samples = samples

    @classmethod
    def from_file(cls, file_handle: BinaryIO, offset: int) -> Chunk:
        raise NotImplementedError(
            "You must call from_file_with_format on a data chunk."
        )

    @classmethod
    def from_file_with_format(
        cls, file_handle: BinaryIO, offset: int, format: FormatChunk
    ) -> Chunk:
        """
        Reads the data chunk from a file with the supplied format.

        Args:
            file_handle (BinaryIO): The file to read in.
            offset (int): The offset to read from.
            format (FormatChunk): The format of the file.

        Returns:
            Chunk: The decoded data chunk.
        """

        # Sanity check

        (header_str, length) = cls.read_header(file_handle, offset)
        if not header_str == cls.HEADER_DATA:
            raise InvalidHeaderException("Data chunk must start with data")

        # Check we have a format

        if not format:
            raise ValueError("You must supply a valid format to read the data chunk as")

        # Create the object

        sample_count = length // format.channels // (format.bits_per_sample // 8)
        samples = np.memmap(
            file_handle,
            dtype=np.dtype(f"<i{format.bits_per_sample // 8}"),
            mode="c",
            shape=(sample_count, format.channels),
        )

        return DataChunk(samples)

    @property
    def samples(self) -> np.ndarray[np.int]:
        """
        The audio sample vectors.
        """
        return self.__samples

    def to_bytes(self) -> List[bytes]:

        # Generate the data section

        data = self.__samples.tobytes()

        # Generate the header

        header = pack("<4sI", self.HEADER_DATA, len(data))

        # Splatter it together

        return b"".join([header, data])

    @property
    def get_name(self) -> str:
        return self.HEADER_DATA
