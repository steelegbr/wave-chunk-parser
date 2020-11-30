from typing import BinaryIO


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
