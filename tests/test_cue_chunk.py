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
from wave_chunk_parser.chunks import CueChunk, CuePoint
from typing import List
from unittest import TestCase


class TestCueChunk(TestCase):
    @parameterized.expand(
        [
            (
                [
                    CuePoint(1, 1, b"data", 0, 0, 12345),
                    CuePoint(2, 2, b"data", 0, 0, 54321),
                ],
                b"cue 4\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00data\x00\x00\x00\x00\x00\x00\x00\x0090\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00data\x00\x00\x00\x00\x00\x00\x00\x001\xd4\x00\x00",
            )
        ]
    )
    def test_encode_cue_chunk(
        self,
        cue_points: List[CuePoint],
        expected: str,
    ):
        """
        Encode a cue chunk.
        """
        cue_chunk = CueChunk(cue_points)
        actual = cue_chunk.to_bytes()
        self.assertEqual(expected, actual)

    @parameterized.expand(
        [
            (
                "./tests/files/cue_chunk.blob",
                0,
                [
                    CuePoint(1, 1, b"data", 0, 0, 12345),
                    CuePoint(2, 2, b"data", 0, 0, 54321),
                ],
            )
        ]
    )
    def test_decode_cue_chunk(
        self, file_name: str, offset: int, expected_cue_points: List[CuePoint]
    ):
        with open(file_name, "rb") as file:
            chunk = CueChunk.from_file(file, offset)
            actual_cue_points = chunk.cue_points

        self.assertIsNotNone(chunk)
        self.assertEqual(len(expected_cue_points), len(actual_cue_points))
        for index, expected_cue_point in enumerate(expected_cue_points):
            self.assertEqual(expected_cue_point.id, actual_cue_points[index].id)
            self.assertEqual(
                expected_cue_point.position, actual_cue_points[index].position
            )
            self.assertEqual(
                expected_cue_point.data_chunk_id, actual_cue_points[index].data_chunk_id
            )
            self.assertEqual(
                expected_cue_point.chunk_start, actual_cue_points[index].chunk_start
            )
            self.assertEqual(
                expected_cue_point.block_start, actual_cue_points[index].block_start
            )
            self.assertEqual(
                expected_cue_point.sample_offset, actual_cue_points[index].sample_offset
            )
