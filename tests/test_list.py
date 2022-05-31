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
from wave_chunk_parser.chunks import Chunk, LabelChunk, ListChunk
from typing import List
from unittest import TestCase


class TestListChunk(TestCase):
    @parameterized.expand(
        [
            (
                [LabelChunk(1, "Label 1"), LabelChunk(2, "Label 2")],
                b"LIST,\x00\x00\x00adtllabl\x0c\x00\x00\x00\x01\x00\x00\x00Label 1\x00labl\x0c\x00\x00\x00\x02\x00\x00\x00Label 2\x00",
            )
        ]
    )
    def test_encode_list(self, sub_chunks: List[Chunk], expected: str):
        """
        Encode a list of labels.
        """
        chunk = ListChunk(ListChunk.HEADER_ASSOC, sub_chunks)
        actual = chunk.to_bytes()
        self.assertEqual(expected, actual)

    @parameterized.expand(
        [
            (
                "./tests/files/list.blob",
                0,
                [LabelChunk(1, "Label 1"), LabelChunk(2, "Label 2")],
            )
        ]
    )
    def test_decode_list(
        self, file_name: str, offset: int, expected_sub_chunks: List[Chunk]
    ):
        """
        List can be decoded successfully
        """
        with open(file_name, "rb") as file:
            chunk = ListChunk.from_file(file, offset)
            actual_sub_chunks = chunk.sub_chunks

        self.assertIsNotNone(chunk)
        self.assertEqual(len(expected_sub_chunks), len(actual_sub_chunks))

        for index, expected_sub_chunk in enumerate(expected_sub_chunks):
            self.assertEqual(expected_sub_chunk.id, actual_sub_chunks[index].id)
            self.assertEqual(expected_sub_chunk.label, actual_sub_chunks[index].label)
