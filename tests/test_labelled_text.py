"""
   Copyright 2024 Marc Steele

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

from wave_chunk_parser.chunks import LabeledTextChunk
from parameterized import parameterized
from unittest import TestCase


class TestLabeledTextChunk(TestCase):
    @parameterized.expand(
        [
            (
                1,
                100,
                "Some Purpose",
                0,
                0,
                0,
                0,
                "Some Label",
                b"ltxt \x00\x00\x00\x01\x00\x00\x00d\x00\x00\x00Some\x00\x00\x00\x00\x00\x00\x00\x00Some Label\x00\x00",
            )
        ]
    )
    def test_encode_labeled_text_chunk(
        self,
        id: int,
        sample_length: int,
        purpose: str,
        country: int,
        language: int,
        dialect: int,
        codepage: int,
        label: str,
        expected_bytes: bytes,
    ):
        """
        A labeled text chunk can be encoded correctly
        """

        # Arrange

        # Act

        chunk = LabeledTextChunk(
            id, sample_length, purpose, country, language, dialect, codepage, label
        )
        actual_bytes = chunk.to_bytes()

        # Assert

        self.assertEqual(actual_bytes, expected_bytes)
