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

from parameterized import parameterized
from wave_chunk_parser.utils import null_terminate, seek_and_read
from unittest import TestCase


class TestFormatChunk(TestCase):
    def test_seek_reject_no_file_handle(self):
        """
        Seek is rejected with no file handle.
        """

        with self.assertRaises(ValueError) as context:
            seek_and_read(None, 0, 0)
            self.assertIn("We must have a stream to read from.", context.exception)

    def test_seek_reject_invalid_start(self):
        """
        Seek is rejected with an invalid start point.
        """

        with self.assertRaises(ValueError) as context:
            seek_and_read(True, -1, 0)
            self.assertIn(
                "Start position for reading cannot be less than zero.",
                context.exception,
            )

    def test_seek_reject_invalid_length(self):
        """
        Seek is rejected with an invalid start point.
        """

        with self.assertRaises(ValueError) as context:
            seek_and_read(True, 0, 0)
            self.assertIn("We need to read at least one byte!", context.exception)

    @parameterized.expand(
        [
            (b"Length 1234", True, b"Length 1234\x00"),
            (b"Length 12345", True, b"Length 12345\x00\x00"),
            (b"Length 1234", False, b"Length 1234\x00"),
            (b"Length 12345", False, b"Length 12345\x00"),
        ]
    )
    def test_null_terminate_expand(
        self, string_to_encode: str, make_even: bool, expected: str
    ):
        """
        Ensure we can null terminate and pad
        """
        actual = null_terminate(string_to_encode, make_even)
        self.assertEqual(expected, actual)
