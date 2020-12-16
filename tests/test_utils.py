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

from wave_chunk_parser.utils import seek_and_read
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
