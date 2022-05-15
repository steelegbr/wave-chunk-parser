"""
   Copyright 2022 Marc Steele

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

from parameterized import parameterized
from wave_chunk_parser.chunks import CuePoint
from typing import List
from unittest import TestCase


class TestCuePoint(TestCase):
    @parameterized.expand(
        [
            (
                1,
                1,
                b"data",
                0,
                0,
                12345,
                b"\x01\x00\x00\x00\x01\x00\x00\x00data\x00\x00\x00\x00\x00\x00\x00\x0090\x00\x00",
            )
        ]
    )
    def test_encode_cue_point(
        self,
        id: int,
        position: int,
        data_chunk_id: int,
        chunk_start: int,
        block_start: int,
        sample_offset: int,
        expected: str,
    ):
        """
        Encode a cue point.
        """
        cue_point = CuePoint(
            id, position, data_chunk_id, chunk_start, block_start, sample_offset
        )
        actual = cue_point.to_bytes()
        self.assertEqual(expected, actual)

    @parameterized.expand(
        [
            (
                "./tests/files/cue.blob",
                0,
                1,
                1,
                b"data",
                0,
                0,
                12345,
            )
        ]
    )
    def test_decode_cue_point(
        self,
        file_name: str,
        offset: int,
        expected_id: int,
        expected_position: int,
        expected_data_chunk_id: str,
        expected_chunk_start: int,
        expected_block_start: int,
        expected_sample_offset: int,
    ):
        with open(file_name, "rb") as file:
            chunk = CuePoint.from_file(file, offset)

        self.assertIsNotNone(chunk)
        self.assertEqual(expected_id, chunk.id)
        self.assertEqual(expected_position, chunk.position)
        self.assertEqual(expected_data_chunk_id, chunk.data_chunk_id)
        self.assertEqual(expected_chunk_start, chunk.chunk_start)
        self.assertEqual(expected_block_start, chunk.block_start)
        self.assertEqual(expected_sample_offset, chunk.sample_offset)
