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
from wave_chunk_parser.chunks import LabelChunk
from unittest import TestCase


class TestLabelChunk(TestCase):
    @parameterized.expand(
        [
            (
                1,
                "The B@nd - 'Funky' S0ng!",
                b"labl\x1e\x00\x00\x00\x01\x00\x00\x00The B@nd - 'Funky' S0ng!\x00\x00",
            )
        ]
    )
    def test_encode_label(self, id: int, label: str, expected: str):
        """
        Label can be encoded successfully
        """
        chunk = LabelChunk(id, label)
        actual = chunk.to_bytes()
        self.assertEqual(expected, actual)

    @parameterized.expand(
        [("./tests/files/label.blob", 0, 1, "The B@nd - 'Funky' S0ng!")]
    )
    def test_decode_label(
        self, file_name: str, offset: int, expected_id: int, expected_label: str
    ):
        """
        Label can be decoded successfully
        """
        with open(file_name, "rb") as file:
            chunk = LabelChunk.from_file(file, offset)

        self.assertIsNotNone(chunk)
        self.assertEqual(expected_id, chunk.id)
        self.assertEqual(expected_label, chunk.label)
