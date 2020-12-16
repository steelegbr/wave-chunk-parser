"""
   Copyright 2020 Marc Steele

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

from wave_chunk_parser.exceptions import (
    ExportExtendedFormatException,
    InvalidHeaderException,
)
from wave_chunk_parser.chunks import FormatChunk, WaveFormat
from parameterized import parameterized
from unittest import TestCase


class TestFormatChunk(TestCase):
    @parameterized.expand(
        [
            (
                "./tests/files/valid_no_markers.wav",
                12,
                WaveFormat.PCM,
                2,
                44100,
                176400,
                4,
                16,
            )
        ]
    )
    def test_read_valid_format_chunk(
        self,
        file_name: str,
        chunk_offset: int,
        expected_format: WaveFormat,
        expected_channels: int,
        expected_sample_rate: int,
        expected_byte_rate: int,
        expected_block_align: int,
        expected_bits_per_sample: int,
    ):
        """
        Valid format chunks can be read from a file.
        """

        # Arrange

        with open(file_name, "rb") as file:

            # Act

            chunk: FormatChunk = FormatChunk.from_file(file, chunk_offset)

            # Assert

            self.assertIsNotNone(chunk)
            self.assertEqual(chunk.get_name, b"fmt ")
            self.assertEqual(chunk.format, expected_format)
            self.assertEqual(chunk.channels, expected_channels)
            self.assertEqual(chunk.sample_rate, expected_sample_rate)
            self.assertEqual(chunk.byte_rate, expected_byte_rate)
            self.assertEqual(chunk.block_align, expected_block_align)
            self.assertEqual(chunk.bits_per_sample, expected_bits_per_sample)
            self.assertFalse(chunk.extended)

    @parameterized.expand(
        [
            (
                "./tests/files/valid_no_markers.wav",
                36,
            )
        ]
    )
    def test_read_wrong_chunk(self, file_name: str, chunk_offset: int):
        """
        An appropriate error is raised if the wrong chunk is read.
        """

        # Arrange

        with open(file_name, "rb") as file:

            # Act

            with self.assertRaises(InvalidHeaderException) as context:
                FormatChunk.from_file(file, chunk_offset)

                # Assert

                self.assertIn("Format chunk must start with fmt", context.exception)

    @parameterized.expand(
        [
            (
                WaveFormat.PCM,
                False,
                2,
                48000,
                16,
                b"fmt \x10\x00\x00\x00\x01\x00\x02\x00\x80\xbb\x00\x00\x00\xee\x02\x00\x04\x00\x10\x00",
            ),
            (
                WaveFormat.A_LAW,
                False,
                1,
                8000,
                8,
                b"fmt \x10\x00\x00\x00\x06\x00\x01\x00@\x1f\x00\x00@\x1f\x00\x00\x01\x00\x08\x00",
            ),
            (
                WaveFormat.MU_LAW,
                False,
                4,
                44100,
                16,
                b"fmt \x10\x00\x00\x00\x07\x00\x04\x00D\xac\x00\x00 b\x05\x00\x08\x00\x10\x00",
            ),
        ]
    )
    def test_encode_chunk(
        self,
        wave_format,
        extended,
        channels,
        sample_rate,
        bits_per_sample,
        expected_bytes: bytes,
    ):
        """
        The format chunk encodes correctly.
        """

        # Arrage

        chunk = FormatChunk(
            wave_format, extended, channels, sample_rate, bits_per_sample
        )

        # Act

        converted = chunk.to_bytes()

        # Assert

        self.assertEqual(converted, expected_bytes)

    def test_fail_encode_extended_format(self):
        """
        We don't successfully encode an extended format chunk.
        """

        # Arrange

        chunk = FormatChunk(WaveFormat.EXTENDED, True, 2, 48000, 16)

        # Act

        with self.assertRaises(ExportExtendedFormatException) as context:
            chunk.to_bytes()

            # Assert

            self.assertIn(
                "We don't support converting extended format headers to binary blobs.",
                context.exception,
            )
