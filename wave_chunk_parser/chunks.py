"""
   Copyright 2020-2022 Marc Steele

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from wave_chunk_parser.exceptions import (
    ExportExtendedFormatException,
    InvalidHeaderException,
    InvalidTimerException,
    InvalidWaveException,
)
from functools import reduce
import numpy as np
from struct import unpack, pack
from typing import BinaryIO, Dict, List, Tuple
from wave_chunk_parser.utils import (
    decode_string,
    encode_string,
    null_terminate,
    seek_and_read,
)


class Chunk(ABC):
    """
    Base class for wave file chunks. We base all other chunks on this.
    """

    OFFSET_CHUNK_CONTENT = 8
    STRUCT_HEADER = "<4sI"

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
            cls.STRUCT_HEADER,
            seek_and_read(file_handle, offset, cls.OFFSET_CHUNK_CONTENT),
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
        cls, file_handle: BinaryIO, offset: int, wave_format: FormatChunk
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

        if not wave_format:
            raise ValueError("You must supply a valid format to read the data chunk as")

        # Read in the raw data

        raw = file_handle.read(length)

        # Create the object

        sample_count = (
            length // wave_format.channels // (wave_format.bits_per_sample // 8)
        )
        samples = np.frombuffer(
            raw,
            dtype=np.dtype(f"<i{wave_format.bits_per_sample // 8}"),
        ).reshape(sample_count, wave_format.channels)

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


class CartTimer:
    """
    A timer associated with a cart.
    """

    __name: str
    __time: int

    __permitted_prefixes = ["SEG", "AUD", "INT", "OUT", "SEC", "TER", "MRK", "EOD"]
    __permitted_start_end = ["SEG", "AUD", "INT", "OUT", "SEC", "TER"]
    __permitted_enumerated = ["SEG", "INT", "OUT", "SEC", "TER", "MRK"]

    def __init__(self, name: str, time: int):

        # Sanity checks

        if not name or not len(name) == 4:
            raise InvalidTimerException("No timer name supplied")

        prefix = name[0:3]
        suffix = name[-1]

        if prefix not in self.__permitted_prefixes:
            raise InvalidTimerException(f"{prefix} is not a valid timer prefix")

        if suffix in ("s", "e") and prefix not in self.__permitted_start_end:
            raise InvalidTimerException(
                f"{prefix} timers cannot have start or end suffixes"
            )

        if suffix.isnumeric() and prefix not in self.__permitted_enumerated:
            raise InvalidTimerException(f"{prefix} timers cannot be enumerated")

        #  If we get this far, we're good to go

        self.__name = name
        self.__time = time

    @classmethod
    def from_cart_parts(cls, timer_parts: List[Tuple[str, int]]) -> List[CartTimer]:
        """
        Builds up a list of timers from given cart parts.

        Args:
            timer_parts (List[Tuple[str, int]]): Tuples of timer name and values.

        Returns:
            List[CartTimer]: The list of valid decoded timers.
        """

        timers = []

        for timer_part in timer_parts:
            try:
                (name, time) = timer_part
                timers.append(CartTimer(decode_string(name), time))
            except InvalidTimerException:
                # Do this to ignore bad timers!
                pass

        return timers

    @property
    def name(self) -> str:
        """
        The name of the timer.
        """
        return self.__name

    @property
    def time(self) -> int:
        """
        The time in samples.
        """
        return self.__time


class CartChunk(Chunk):
    """
    Broadcast cart chunk.
    """

    LENGTH_MINIMUM = 2048
    HEADER_CART = b"cart"
    DEFAULT_VERSION = b"0101"
    FORMAT_DATE_TIME = "%Y/%m/%d%H:%M:%S"
    UNPACK_STRING = (
        "<4s64s64s64s64s64s64s64s18s18s64s64s64si4sI4sI4sI4sI4sI4sI4sI4sI276s1024s"
    )
    PACK_STRING = (
        "<4sI4s64s64s64s64s64s64s64s18s18s64s64s64si4sI4sI4sI4sI4sI4sI4sI4sI276s1024s"
    )

    __version: str
    __title: str
    __artist: str
    __cut_id: str
    __client_id: str
    __category: str
    __classification: str
    __out_cue: str
    __start_date: datetime
    __end_date: datetime
    __producer_app: str
    __producer_app_version: str
    __user_defined: str
    __ref_0db: int
    __timers: List[CartTimer]
    __url: str
    __tag_text: str

    def __init__(
        self,
        version: str,
        title: str,
        artist: str,
        cut_id: str,
        client_id: str,
        category: str,
        classification: str,
        out_cue: str,
        start_date: datetime,
        end_date: datetime,
        producer_app: str,
        producer_app_version: str,
        user_defined: str,
        ref_0db: int,
        timers: List[CartTimer],
        url: str,
        tag_text: str,
    ):
        self.__version = version
        self.__title = title
        self.__artist = artist
        self.__cut_id = cut_id
        self.__client_id = client_id
        self.__category = category
        self.__classification = classification
        self.__out_cue = out_cue
        self.__start_date = start_date
        self.__end_date = end_date
        self.__producer_app = producer_app
        self.__producer_app_version = producer_app_version
        self.__user_defined = user_defined
        self.__ref_0db = ref_0db
        self.__timers = timers
        self.__url = url
        self.__tag_text = tag_text

    @classmethod
    def from_file(cls, file_handle: BinaryIO, offset: int) -> Chunk:

        # Sanity checks

        (header_str, length) = cls.read_header(file_handle, offset)
        if not header_str == cls.HEADER_CART:
            raise InvalidHeaderException("Cart chunk must start with cart")

        if length < cls.LENGTH_MINIMUM:
            raise InvalidHeaderException(
                f"Cart chunk is not long enough. Must be a minimum of {cls.LENGTH_MINIMUM} bytes"
            )

        # Read from the chunk

        tag_text = None
        unpack_string = cls.UNPACK_STRING

        if length > cls.LENGTH_MINIMUM:
            unpack_string += f"{length - cls.LENGTH_MINIMUM}s"
            (
                version,
                title,
                artist,
                cut_id,
                client_id,
                category,
                classification,
                out_cue,
                start_date_str,
                end_date_str,
                producer_app_id,
                producer_app_version,
                user_defined,
                ref_0db,
                timer_name_0,
                timer_time_0,
                timer_name_1,
                timer_time_1,
                timer_name_2,
                timer_time_2,
                timer_name_3,
                timer_time_3,
                timer_name_4,
                timer_time_4,
                timer_name_5,
                timer_time_5,
                timer_name_6,
                timer_time_6,
                timer_name_7,
                timer_time_7,
                _,
                url,
                tag_text,
            ) = unpack(
                unpack_string,
                seek_and_read(
                    file_handle,
                    offset + cls.OFFSET_CHUNK_CONTENT,
                    length,
                ),
            )
        else:
            (
                version,
                title,
                artist,
                cut_id,
                client_id,
                category,
                classification,
                out_cue,
                start_date_str,
                end_date_str,
                producer_app_id,
                producer_app_version,
                user_defined,
                ref_0db,
                timer_name_0,
                timer_time_0,
                timer_name_1,
                timer_time_1,
                timer_name_2,
                timer_time_2,
                timer_name_3,
                timer_time_3,
                timer_name_4,
                timer_time_4,
                timer_name_5,
                timer_time_5,
                timer_name_6,
                timer_time_6,
                timer_name_7,
                timer_time_7,
                _,
                url,
            ) = unpack(
                unpack_string,
                seek_and_read(
                    file_handle,
                    offset + cls.OFFSET_CHUNK_CONTENT,
                    length,
                ),
            )

        # Extract out the timers

        timers = CartTimer.from_cart_parts(
            [
                (timer_name_0, timer_time_0),
                (timer_name_1, timer_time_1),
                (timer_name_2, timer_time_2),
                (timer_name_3, timer_time_3),
                (timer_name_4, timer_time_4),
                (timer_name_5, timer_time_5),
                (timer_name_6, timer_time_6),
                (timer_name_7, timer_time_7),
            ]
        )

        # Decode the start/end datetime

        start_date = datetime.strptime(
            decode_string(start_date_str), cls.FORMAT_DATE_TIME
        )
        end_date = datetime.strptime(decode_string(end_date_str), cls.FORMAT_DATE_TIME)

        # Build the chunk

        return CartChunk(
            decode_string(version),
            decode_string(title),
            decode_string(artist),
            decode_string(cut_id),
            decode_string(client_id),
            decode_string(category),
            decode_string(classification),
            decode_string(out_cue),
            start_date,
            end_date,
            decode_string(producer_app_id),
            decode_string(producer_app_version),
            decode_string(user_defined),
            ref_0db,
            timers,
            decode_string(url),
            decode_string(tag_text),
        )

    @property
    def get_name(self) -> str:
        return self.HEADER_CART

    def to_bytes(self) -> List[bytes]:

        # Explode out our list of timers

        timer_values = [
            b"",
            0,
            b"",
            0,
            b"",
            0,
            b"",
            0,
            b"",
            0,
            b"",
            0,
            b"",
            0,
            b"",
            0,
        ]
        timers = self.timers[0:8]

        for index, timer in enumerate(timers):
            timer_values[index * 2] = encode_string(timer.name)
            timer_values[index * 2 + 1] = timer.time

        # Work out the chunk length

        pack_string = self.PACK_STRING
        length = self.LENGTH_MINIMUM

        if self.tag_text:
            tag_text_length = len(self.tag_text)
            length += tag_text_length
            pack_string += f"{tag_text_length}s"
            return pack(
                pack_string,
                self.HEADER_CART,
                length,
                encode_string(self.version),
                encode_string(self.title),
                encode_string(self.artist),
                encode_string(self.cut_id),
                encode_string(self.client_id),
                encode_string(self.category),
                encode_string(self.classification),
                encode_string(self.out_cue),
                encode_string(self.start_date.strftime(self.FORMAT_DATE_TIME)),
                encode_string(self.end_date.strftime(self.FORMAT_DATE_TIME)),
                encode_string(self.producer_app),
                encode_string(self.producer_app_version),
                encode_string(self.user_defined),
                self.ref_0db,
                timer_values[0],
                timer_values[1],
                timer_values[2],
                timer_values[3],
                timer_values[4],
                timer_values[5],
                timer_values[6],
                timer_values[7],
                timer_values[8],
                timer_values[9],
                timer_values[10],
                timer_values[11],
                timer_values[12],
                timer_values[13],
                timer_values[14],
                timer_values[15],
                b"",
                encode_string(self.url),
                encode_string(self.tag_text),
            )

        return pack(
            pack_string,
            self.HEADER_CART,
            length,
            encode_string(self.version),
            encode_string(self.title),
            encode_string(self.artist),
            encode_string(self.cut_id),
            encode_string(self.client_id),
            encode_string(self.category),
            encode_string(self.classification),
            encode_string(self.out_cue),
            encode_string(self.start_date.strftime(self.FORMAT_DATE_TIME)),
            encode_string(self.end_date.strftime(self.FORMAT_DATE_TIME)),
            encode_string(self.producer_app),
            encode_string(self.producer_app_version),
            encode_string(self.user_defined),
            self.ref_0db,
            timer_values[0],
            timer_values[1],
            timer_values[2],
            timer_values[3],
            timer_values[4],
            timer_values[5],
            timer_values[6],
            timer_values[7],
            timer_values[8],
            timer_values[9],
            timer_values[10],
            timer_values[11],
            timer_values[12],
            timer_values[13],
            timer_values[14],
            timer_values[15],
            b"",
            encode_string(self.url),
        )

    @property
    def version(self) -> str:
        """
        The cart chunk version. Usually 0101.
        """
        return self.__version

    @property
    def title(self) -> str:
        """
        The cart title.
        """
        return self.__title

    @property
    def artist(self) -> str:
        """
        The artist of the audio on the cart.
        """
        return self.__artist

    @property
    def cut_id(self) -> str:
        """
        The unique cut ID. Not to be confused with the cart ID.
        """
        return self.__cut_id

    @property
    def client_id(self) -> str:
        """
        The client ID. Mainly used for adverts.
        """
        return self.__client_id

    @property
    def category(self) -> str:
        """
        The category of the cart. While this is a freeform text field, AES46-2002 has a list of recommendations.
        """
        return self.__category

    @property
    def classification(self) -> str:
        """
        Another field we can use of categorisation.
        """
        return self.__classification

    @property
    def out_cue(self) -> str:
        """
        The out cue for any presenter / journalist using the cart. Not seen often in the wild.
        """
        return self.__out_cue

    @property
    def start_date(self) -> datetime:
        """
        The valid start date/time.
        """
        return self.__start_date

    @property
    def end_date(self) -> datetime:
        """
        The valid end date/time.
        """
        return self.__end_date

    @property
    def producer_app(self) -> str:
        """
        The name of the application that produced the cart.
        """
        return self.__producer_app

    @property
    def producer_app_version(self) -> str:
        """
        The version of the application that produced the cart.
        """
        return self.__producer_app_version

    @property
    def user_defined(self) -> str:
        """
        A user defined text field.
        """
        return self.__user_defined

    @property
    def ref_0db(self) -> int:
        """
        The 0dB reference level.
        """
        return self.__ref_0db

    @property
    def timers(self) -> List[CartTimer]:
        """
        The timers / markers for the audio. This is the bit playout systems use.
        """
        return self.__timers

    @property
    def url(self) -> str:
        """
        A URL linked from the audio file.
        """
        return self.__url

    @property
    def tag_text(self) -> str:
        """
        A freeform text field. Used by Master Control and friends to store extra metadata in XML, JSON, etc.
        """
        return self.__tag_text


class LabelChunk(Chunk):
    """
    A label associated with a cue point
    """

    HEADER_LABEL = b"labl"
    OFFSET_LABEL = 4
    OFFSET_HEADER = 4

    __id: int
    __label: str

    def __init__(self, id: int, label: str):
        self.__id = id
        self.__label = label

    @property
    def get_name(self) -> str:
        return self.HEADER_LABEL  # skipcq: TCV-001

    def to_bytes(self) -> List[bytes]:
        encoded_label = null_terminate(encode_string(self.__label), True)
        encoded_label_length = len(encoded_label)

        return pack(
            f"<4sII{encoded_label_length}s",
            self.HEADER_LABEL,
            encoded_label_length + self.OFFSET_LABEL,
            self.__id,
            encoded_label,
        )

    @classmethod
    def from_file(cls, file_handle: BinaryIO, offset: int) -> Chunk:
        # Sanity checks

        (header_str, length) = cls.read_header(file_handle, offset)
        if not header_str == cls.HEADER_LABEL:
            raise InvalidHeaderException(
                "Label header must start with label"
            )  # skipcq: TCV-001

        # Read the rest of the header

        new_id, raw_label = unpack(
            f"<I{length - cls.OFFSET_LABEL}s",
            seek_and_read(
                file_handle,
                offset + cls.OFFSET_CHUNK_CONTENT,
                length,
            ),
        )
        return LabelChunk(new_id, decode_string(raw_label))

    @property
    def id(self) -> str:
        """
        The cue point ID this label is for.
        """
        return self.__id

    @property
    def label(self) -> str:
        """
        The label value.
        """
        return self.__label


class ListChunk(Chunk):
    """
    Associated data list chunk. In this implementation, limited to labl children.
    """

    __sub_chunks: List[Chunk] = []

    HEADER_LIST = b"LIST"
    HEADER_ASSOC = b"adtl"
    LENGTH_ASSOC = 4
    OFFSET_SUBCHUNKS = 12
    CHUNK_HEADER_MAP = {
        b"labl": LabelChunk,
    }

    def __init__(self, sub_chunks: List[Chunk]):
        self.__sub_chunks = sub_chunks

    @classmethod
    def from_file(cls, file_handle: BinaryIO, offset: int) -> Chunk:

        # Sanity checks

        (header_str, length) = cls.read_header(file_handle, offset)
        if not header_str == cls.HEADER_LIST:
            raise InvalidHeaderException(
                "List header must start with list"
            )  # skipcq: TCV-001

        # Read in the sub chunks

        current_offset = offset + cls.OFFSET_SUBCHUNKS
        end_of_chunk = offset + length
        sub_chunks = []

        while current_offset < end_of_chunk:
            (current_header, current_length) = cls.read_header(
                file_handle, current_offset
            )
            chunk_type = cls.CHUNK_HEADER_MAP.get(current_header)

            if chunk_type:
                current_sub_chunk = chunk_type.from_file(file_handle, current_offset)
                sub_chunks.append(current_sub_chunk)

            current_offset += current_length + cls.OFFSET_CHUNK_CONTENT

        return ListChunk(sub_chunks)

    def to_bytes(self) -> List[bytes]:
        encoded_sub_chunks = [sub_chunk.to_bytes() for sub_chunk in self.sub_chunks]
        combined_sub_chunks = b"".join(encoded_sub_chunks)
        length = len(combined_sub_chunks)

        header = pack(
            "<4sI4s", self.HEADER_LIST, length + self.LENGTH_ASSOC, self.HEADER_ASSOC
        )
        return b"".join([header, combined_sub_chunks])

    @property
    def sub_chunks(self) -> List[Chunk]:
        """
        Sub chunks in the list
        """
        return self.__sub_chunks

    @property
    def get_name(self) -> str:
        return self.HEADER_LIST  # skipcq: TCV-001


class CuePoint(Chunk):
    """
    An individual cue point.
    """

    __id: int
    __position: int
    __data_chunk_id: str
    __chunk_start: int
    __block_start: int
    __sample_offset: int

    LENGTH_CUE = 24

    def __init__(
        self,
        id: int,
        position: int,
        data_chunk_id: str,
        chunk_start: int,
        block_start: int,
        sample_offset: int,
    ):
        self.__id = id
        self.__position = position
        self.__data_chunk_id = data_chunk_id
        self.__chunk_start = chunk_start
        self.__block_start = block_start
        self.__sample_offset = sample_offset

    @classmethod
    def from_file(cls, file_handle: BinaryIO, offset: int) -> CuePoint:
        (id, position, data_chunk_id, chunk_start, block_start, sample_offset) = unpack(
            "<II4sIII", seek_and_read(file_handle, offset, cls.LENGTH_CUE)
        )
        return CuePoint(
            id, position, data_chunk_id, chunk_start, block_start, sample_offset
        )

    @property
    def id(self) -> int:
        """
        The unique cue point identifier
        """
        return self.__id

    @property
    def position(self) -> int:
        """
        The position of the cue point.
        """
        return self.__position

    @property
    def data_chunk_id(self) -> str:
        """
        The data chunk number this is from. Our simple implementation always assumes b"data".
        """
        return self.__data_chunk_id

    @property
    def chunk_start(self) -> int:
        """
        Where the chunk starts this cue point is for. Again, we assume 0.
        """
        return self.__chunk_start

    @property
    def block_start(self) -> int:
        """
        The byte offset to start looking in the block. We assume 0 here.
        """
        return self.__block_start

    @property
    def sample_offset(self) -> int:
        """
        The sample number the cue point matches. This is the same behaviour as cart chunk.
        """
        return self.__sample_offset

    @property
    def get_name(self) -> str:
        pass  # skipcq: TCV-001

    def to_bytes(self) -> List[bytes]:
        return pack(
            "<II4sIII",
            self.id,
            self.position,
            self.data_chunk_id,
            self.chunk_start,
            self.block_start,
            self.sample_offset,
        )


class CueChunk(Chunk):
    """
    A list of cue points
    """

    __cue_points: List[CuePoint] = []

    HEADER_CUE = b"cue "
    OFFSET_CUE_COUNT = 8
    OFFSET_CUE_POINTS = 12
    LENGTH_CUE_POINT = 24
    LENGTH_CUE_COUNT = 4

    def __init__(self, cue_points: List[CuePoint]) -> CueChunk:
        self.__cue_points = cue_points

    @classmethod
    def from_file(cls, file_handle: BinaryIO, offset: int) -> CueChunk:

        # Sanity checks

        (header_str, length) = cls.read_header(file_handle, offset)
        if not header_str == cls.HEADER_CUE:
            raise InvalidHeaderException(
                "Cue point chunk must start with cue"
            )  # skipcq: TCV-001

        # Read from the chunk

        (sub_chunk_count,) = unpack(
            "<I",
            seek_and_read(
                file_handle, offset + cls.OFFSET_CUE_COUNT, cls.LENGTH_CUE_COUNT
            ),
        )
        if length != sub_chunk_count * cls.LENGTH_CUE_POINT + cls.LENGTH_CUE_COUNT:
            raise InvalidHeaderException(
                f"Cue chunk length of {length} does not match for {sub_chunk_count} cue points"
            )  # skipcq: TCV-001

        cue_points = []
        current_offset = offset + cls.OFFSET_CUE_POINTS
        end_of_chunk = offset + length

        while current_offset < end_of_chunk:
            current_cue_point = CuePoint.from_file(file_handle, current_offset)
            cue_points.append(current_cue_point)
            current_offset += cls.LENGTH_CUE_POINT

        return CueChunk(cue_points)

    @property
    def get_name(self) -> str:
        return self.HEADER_CUE  # skipcq: TCV-001

    @property
    def cue_points(self) -> List[CuePoint]:
        return self.__cue_points

    def to_bytes(self) -> List[bytes]:
        encoded_cue_points = [cue_point.to_bytes() for cue_point in self.cue_points]
        cue_point_count = len(self.cue_points)
        length = cue_point_count * self.LENGTH_CUE_POINT + self.LENGTH_CUE_COUNT

        header = pack("<4sII", self.HEADER_CUE, length, cue_point_count)
        return b"".join([header, *encoded_cue_points])


class RiffChunk(Chunk):
    """
    The second level WAVE chunk in a RIFF file.
    """

    __sub_chunks: Dict[str, Chunk] = {}

    HEADER_RIFF = b"RIFF"
    HEADER_WAVE = b"WAVE"
    CHUNK_HEADER_MAP = {
        b"fmt ": FormatChunk,
        b"data": DataChunk,
        b"cart": CartChunk,
        b"LIST": ListChunk,
        b"cue ": CueChunk,
    }

    CHUNK_FORMAT = b"fmt "
    CHUNK_DATA = b"data"
    CHUNK_CART = b"cart"
    CHUNK_CUE = b"cue "
    CHUNK_LIST = b"LIST"

    OFFSET_SUB_TYPE = 8
    OFFSET_CHUNKS_START = 12

    LENGTH_SUB_TYPE = 4

    STRUCT_SUB_TYPE = "<4s"
    STRUCT_RIFF_HEADER = "<4sI4s"

    def __init__(self, sub_chunks: Dict[str, Chunk]) -> None:
        self.__sub_chunks = sub_chunks

    @classmethod
    def from_file(cls, file_handle: BinaryIO, offset: int = 0) -> Chunk:

        # Sanity check

        (header_str, length) = cls.read_header(file_handle, offset)

        if not header_str == cls.HEADER_RIFF:
            raise InvalidHeaderException("WAVE files must have a RIFF header")

        if not length:
            raise InvalidHeaderException(
                "WAVE files must have a length greater than zero"
            )

        # Check the RIFF sub-type

        (sub_type,) = unpack(
            cls.STRUCT_SUB_TYPE,
            seek_and_read(
                file_handle, offset + cls.OFFSET_SUB_TYPE, cls.LENGTH_SUB_TYPE
            ),
        )
        if not sub_type == cls.HEADER_WAVE:
            raise InvalidHeaderException("This library only supports WAVE files")

        # Read in the sub-chunks

        current_offset = offset + cls.OFFSET_CHUNKS_START
        file_handle.seek(0, 2)
        end_of_file = min(file_handle.tell(), length + current_offset)
        chunk_map = {}

        while current_offset < end_of_file:
            (current_header, current_length) = cls.read_header(
                file_handle, current_offset
            )
            chunk_type = cls.CHUNK_HEADER_MAP.get(current_header)

            if chunk_type:
                if chunk_type == DataChunk:
                    if b"fmt " not in chunk_map:
                        raise InvalidWaveException(
                            "A format chunk must be read before a data chunk!"
                        )
                    chunk = DataChunk.from_file_with_format(
                        file_handle, current_offset, chunk_map.get(cls.CHUNK_FORMAT)
                    )
                else:
                    chunk = chunk_type.from_file(file_handle, current_offset)

                chunk_map[current_header] = chunk

            # Cycle onto the next chunk

            current_offset += current_length + cls.OFFSET_CHUNK_CONTENT

        return RiffChunk(chunk_map)

    @property
    def sub_chunks(self) -> Dict[str, Chunk]:
        return self.__sub_chunks

    @property
    def get_name(self) -> str:
        return self.HEADER_WAVE

    def to_bytes(self) -> List[bytes]:

        # Check we have at least a format and data chunk

        if self.CHUNK_FORMAT not in self.sub_chunks:
            raise InvalidWaveException("Valid wave files must have a format chunk")

        if self.CHUNK_DATA not in self.sub_chunks:
            raise InvalidWaveException("Valid wave files must have a data chunk")

        # Build our chunks up in order

        chunk_bytes = []
        chunk_bytes.append(self.sub_chunks.get(self.CHUNK_FORMAT).to_bytes())

        if self.CHUNK_CART in self.sub_chunks:
            chunk_bytes.append(self.sub_chunks.get(self.CHUNK_CART).to_bytes())

        if self.CHUNK_LIST in self.sub_chunks:
            chunk_bytes.append(self.sub_chunks.get(self.CHUNK_LIST).to_bytes())

        if self.CHUNK_CUE in self.sub_chunks:
            chunk_bytes.append(self.sub_chunks.get(self.CHUNK_CUE).to_bytes())

        chunk_bytes.append(self.sub_chunks.get(self.CHUNK_DATA).to_bytes())

        # Create the header

        lengths = [len(chunk) for chunk in chunk_bytes]
        length = reduce(lambda a, b: a + b, lengths) + self.LENGTH_SUB_TYPE
        header = pack(
            self.STRUCT_RIFF_HEADER, self.HEADER_RIFF, length, self.HEADER_WAVE
        )
        chunk_bytes.insert(0, header)

        # Create a blob

        return b"".join(chunk_bytes)
