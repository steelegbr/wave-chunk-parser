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

import numpy as np

from io import BytesIO
from parameterized import parameterized
from typing import Tuple
from unittest import TestCase
from wave_chunk_parser.chunks import DataChunk, FormatChunk, WaveFormat
from wave_chunk_parser.exceptions import InvalidHeaderException


class TestDataChunk(TestCase):
    @parameterized.expand(
        [
            (
                "./tests/files/valid_no_markers.wav",
                36,
                FormatChunk(WaveFormat.PCM, False, 2, 44100, 16),
                (111020, 2),
            )
        ]
    )
    def test_read_valid_data_chunk(
        self,
        file_name: str,
        chunk_offset: int,
        wave_format: FormatChunk,
        expected_shape: Tuple[int, int],
    ):
        """
        The data chunk can be read correctly.
        """

        #  Arrange

        with open(file_name, "rb") as file:

            # Act

            chunk: DataChunk = DataChunk.from_file_with_format(
                file, chunk_offset, wave_format
            )
            samples = chunk.samples

            # Assert

            self.assertIsNotNone(chunk)
            self.assertEqual(chunk.get_name, b"data")
            self.assertEqual(samples.shape, expected_shape)

    @parameterized.expand(
        [
            (
                "./tests/files/valid_no_markers.wav",
                36,
                FormatChunk(WaveFormat.PCM, False, 2, 44100, 16),
                (111020, 2),
            )
        ]
    )
    def test_read_valid_data_chunk_as_blob(
        self,
        file_name: str,
        chunk_offset: int,
        wave_format: FormatChunk,
        expected_shape: Tuple[int, int],
    ):
        """
        The data chunk can be read correctly from a blob.
        """

        #  Arrange

        with open(file_name, "rb") as file:
            blob = file.read()

        file = BytesIO(blob)

        # Act

        chunk: DataChunk = DataChunk.from_file_with_format(
            file, chunk_offset, wave_format
        )
        samples = chunk.samples

        # Assert

        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.get_name, b"data")
        self.assertEqual(samples.shape, expected_shape)

    @parameterized.expand(
        [
            (
                "./tests/files/valid_no_markers.wav",
                12,
                FormatChunk(WaveFormat.PCM, False, 2, 44100, 16),
            )
        ]
    )
    def test_read_wrong_chunk(
        self, file_name: str, chunk_offset: int, wave_format: FormatChunk
    ):
        """
        An appropriate error is raised if the wrong chunk is read.
        """

        # Arrange

        with open(file_name, "rb") as file:

            # Act

            with self.assertRaises(InvalidHeaderException) as context:
                DataChunk.from_file_with_format(file, chunk_offset, wave_format)

                # Assert

                self.assertIn("Data chunk must start with data", context.exception)

    @parameterized.expand(
        [
            (
                "./tests/files/valid_no_markers.wav",
                36,
            )
        ]
    )
    def test_read_no_format(self, file_name: str, chunk_offset: int):
        """
        No format information raises an error.
        """

        # Arrange

        with open(file_name, "rb") as file:

            # Act

            with self.assertRaises(ValueError) as context:
                DataChunk.from_file_with_format(file, chunk_offset, None)

                # Assert

                self.assertIn(
                    "You must supply a valid format to read the data chunk as",
                    context.exception,
                )

    @parameterized.expand(
        [
            (
                np.ndarray(
                    shape=(3, 2),
                    dtype=np.dtype("<i2"),
                    buffer=np.array([-32768, -32768, 0, 0, 32767, 32767]),
                ),
                b"data\x0c\x00\x00\x00\x00\x80\xff\xff\xff\xff\xff\xff\x00\x80\xff\xff",
            )
        ]
    )
    def test_encode_chunk(self, samples: np.ndarray, expected_result: bytes):
        """
        Encode a data chunk.
        """

        # Arrange

        chunk = DataChunk(samples)

        # Act

        chunk_bytes = chunk.to_bytes()

        # Assert

        self.assertIsNotNone(chunk_bytes)
        self.assertEqual(chunk_bytes, expected_result)

    def test_from_file_raises_exception(self):
        """
        Calling from_file points us to use the correct function.
        """

        with self.assertRaises(NotImplementedError) as context:
            DataChunk.from_file(None, 0)
            self.assertIn(
                "You must call from_file_with_format on a data chunk.",
                context.exception,
            )
