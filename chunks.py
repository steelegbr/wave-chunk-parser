from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from exceptions import ExportExtendedFormatException, InvalidHeaderException
from typing import BinaryIO, List
from utils import seek_and_read, seek_and_read_int


class Chunk(ABC):
    """
    Base class for wave file chunks. We base all other chunks on this.
    """

    LENGTH_HEADER = 4
    LENGTH_LENGTH = 4
    OFFSET_LENGTH = 4

    @property
    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the chunk type.
        """

        raise NotImplementedError

    @abstractmethod
    def to_bytes(self) -> List[bytes]:
        """
        Encodes the chunck to a byte array for writing to a file.

        Returns:
            List[bytes]: The encoded chunk.
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_file(
        cls, file_handle: BinaryIO, offset: int, endiness: str = "little"
    ) -> Chunk:
        """
        Creates a chunk from a file.

        Args:
            file_handle (BinaryIO): The file handle to use.
            offset (int): The offset in the file this chunk is at.
            endiness (str): "big" or "small" - the endianess of the file.

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

    LENGTH_FORMAT_CODE = 2
    LENGTH_CHANNELS = 2
    LENGTH_SAMPLE_RATE = 2
    LENGTH_BYTE_RATE = 2
    LENGTH_BLOCK_ALIGN = 2
    LENGTH_BITS_PER_SAMPLE = 2
    LENGTH_STANDARD_SIZE = 16
    OFFSET_FORMAT_CODE = 8
    OFFSET_CHANNELS = 10
    OFFSET_SAMPLE_RATE = 12
    OFFSET_BYTE_RATE = 16
    OFFSET_BLOCK_ALIGN = 20
    OFFSET_BITS_PER_SAMPLE = 22

    HEADER_FORMAT = "fmt "

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
    def from_file(
        cls, file_handle: BinaryIO, offset: int, endiness: str = "little"
    ) -> FormatChunk:

        # Sanity check

        header_bytes = seek_and_read(file_handle, offset, cls.LENGTH_HEADER)
        header_str = header_bytes.decode("ASCII")
        if not header_str == cls.HEADER_FORMAT:
            raise InvalidHeaderException("Format chunk must start with fmt")

        # Check the length

        length = seek_and_read_int(
            file_handle, offset + cls.OFFSET_LENGTH, cls.LENGTH_LENGTH, endiness
        )
        extended = length > cls.LENGTH_STANDARD_SIZE

        # Read the format

        handle = seek_and_read_int(
            file_handle,
            offset + cls.OFFSET_FORMAT_CODE,
            cls.LENGTH_FORMAT_CODE,
            endiness,
        )
        wave_format = WaveFormat(handle)

        # Number of channels, sample rate, etc.

        channels = seek_and_read_int(
            file_handle, offset + cls.OFFSET_CHANNELS, cls.LENGTH_CHANNELS, endiness
        )
        sample_rate = seek_and_read_int(
            file_handle,
            offset + cls.OFFSET_SAMPLE_RATE,
            cls.LENGTH_SAMPLE_RATE,
            endiness,
        )
        bits_per_sample = seek_and_read_int(
            file_handle,
            offset + cls.OFFSET_BITS_PER_SAMPLE,
            cls.LENGTH_BITS_PER_SAMPLE,
            endiness,
        )

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
            raise ExportExtendedFormatException("We don't support converting extended format headers to binary blobs.")

        # Start with the header

        chunk = []

        chunk.append(self.HEADER_FORMAT.encode("ASCII"))
        chunk.append(bytes(self.LENGTH_STANDARD_SIZE))
        chunk.append(bytes(self.format.value))
        chunk.append(bytes(self.channels))
        chunk.append(bytes(self.sample_rate))
        chunk.append(bytes(self.byte_rate))
        chunk.append(bytes(self.block_align))
        chunk.append(bytes(self.bits_per_sample))

        return b"".join(chunk)

