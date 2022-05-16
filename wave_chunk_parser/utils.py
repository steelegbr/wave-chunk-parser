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

from typing import BinaryIO
from unidecode import unidecode


def seek_and_read(file_handle: BinaryIO, start: int, length: int) -> bytes:
    """
    Seeks and reads a byte array from a stream.

    Args:
        file_handle (BinaryIO): The stream to read from.
        start (int): The start position.
        length (int): The number of bytes to attempt to read.

    Returns:
        bytes: The bytes requested.
    """

    # Sanity check

    if not file_handle:
        raise ValueError("We must have a stream to read from.")

    if start < 0:
        raise ValueError("Start position for reading cannot be less than zero.")

    if length < 1:
        raise ValueError("We need to read at least one byte!")

    # Perform the read

    file_handle.seek(start)
    return file_handle.read(length)


def decode_string(byte_string: bytes, encoding: str = "ASCII") -> str:
    """
    Decodes a byte string to a given encoding.

    Args:
        byte_string (bytes): The byte string to decode.
        encoding (str, optional): The encoding to use. Defaults to "ASCII".

    Returns:
        str: The decoded byte string.
    """

    if not byte_string:
        return None

    decoded = byte_string.decode(encoding)
    return decoded.replace("\x00", "")


def encode_string(string: str, encoding: str = "ASCII") -> str:
    """
    Encode a byte string.

    Args:
        string (str): The string to encode.
        encoding (str, optional): The encoding to use.. Defaults to "ASCII".

    Returns:
        str: The encoded byte string.
    """

    unidecoded = unidecode(string)
    return unidecoded.encode(encoding)


def null_terminate(string: str, make_even_length=False) -> str:
    """
    Null terminates a string. Optionally makes the string even length.
    """

    terminated = string + b"\x00"
    if make_even_length and len(terminated) % 2:
        return terminated + b"\x00"
    return terminated
