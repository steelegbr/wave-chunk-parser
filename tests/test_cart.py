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

from wave_chunk_parser.chunks import CartChunk, CartTimer
from datetime import datetime
from wave_chunk_parser.exceptions import InvalidHeaderException, InvalidTimerException
import json
from parameterized import parameterized
from typing import List, Tuple
from unittest import TestCase


class TestCartChunk(TestCase):
    @parameterized.expand(
        [
            (
                "./tests/files/cart_no_tag.blob",
                0,
                "0101",
                "A cart with no tag text",
                "Some artist",
                "TAGLESS",
                "Free Spirit",
                "DEMO",
                "Demo and sample files",
                "Nom Nom Nom!",
                datetime(1900, 1, 1, 0, 0),
                datetime(2099, 12, 31, 23, 59, 59),
                "Hand Crafted",
                "MK1 Eyeball",
                "Some stuff goes in here....",
                32768,
                [("MRK ", 112000), ("SEC1", 152533), ("EOD", 201024)],
                "http://www.example.com/",
                None,
            ),
            (
                "./tests/files/cart_long.blob",
                0,
                "0101",
                "This is a cart with a really long title that should be trunkated",
                "This is a cart with a really long artist name that should be tru",
                "LONGCART",
                "Biscuit Muncher",
                "DEMO",
                "Demo and sample files",
                "Nom Nom Nom!",
                datetime(1900, 1, 1, 0, 0),
                datetime(2099, 12, 31, 23, 59, 59),
                "Hand Crafted",
                "MK1 Eyeball",
                "Some stuff goes in here....",
                32768,
                [("MRK ", 112000), ("SEC1", 152533), ("EOD", 201024)],
                "http://www.example.com/",
                "A load of junk goes in here.\r\n",
            ),
        ]
    )
    def test_read_valid_data_chunk(
        self,
        file_name: str,
        chunk_offset: int,
        expected_version: str,
        expected_title: str,
        expected_artist: str,
        expected_cut_id: str,
        expected_client_id: str,
        expected_category: str,
        expected_classification: str,
        expected_out_cue: str,
        expected_start_date: datetime,
        expected_end_date: datetime,
        expected_producer_app: str,
        expected_producer_app_version: str,
        expected_user_defined: str,
        expected_ref_0db: int,
        expected_timers: List[Tuple[str, int]],
        expected_url: str,
        expected_tag_text: str,
    ):
        """
        The cart chunk can be read correctly.
        """

        #  Arrange

        with open(file_name, "rb") as file:

            # Act

            chunk = CartChunk.from_file(file, chunk_offset)

            # Assert

            self.assertIsNotNone(chunk)
            self.assertEqual(chunk.get_name, b"cart")
            self.assertEqual(chunk.version, expected_version)
            self.assertEqual(chunk.title, expected_title)
            self.assertEqual(chunk.artist, expected_artist)
            self.assertEqual(chunk.cut_id, expected_cut_id)
            self.assertEqual(chunk.client_id, expected_client_id)
            self.assertEqual(chunk.category, expected_category)
            self.assertEqual(chunk.classification, expected_classification)
            self.assertEqual(chunk.out_cue, expected_out_cue)
            self.assertEqual(chunk.start_date, expected_start_date)
            self.assertEqual(chunk.end_date, expected_end_date)
            self.assertEqual(chunk.producer_app, expected_producer_app)
            self.assertEqual(chunk.producer_app_version, expected_producer_app_version)
            self.assertEqual(chunk.user_defined, expected_user_defined)
            self.assertEqual(chunk.ref_0db, expected_ref_0db)
            self.assertEqual(len(chunk.timers), len(expected_timers))

            for (expected_name, expected_time) in expected_timers:
                self.assertTrue(
                    [
                        timer.name == expected_name and timer.time == expected_time
                        for timer in chunk.timers
                    ]
                )

            self.assertEqual(chunk.url, expected_url)
            self.assertEqual(chunk.tag_text, expected_tag_text)

    @parameterized.expand(
        [
            (
                "./tests/files/valid_no_markers.wav",
                12,
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
                CartChunk.from_file(file, chunk_offset)

                # Assert

                self.assertIn("Cart chunk must start with cart", context.exception)

    @parameterized.expand(
        [
            ("./tests/files/cart_long.json", "./tests/files/cart_long.blob"),
            ("./tests/files/cart_no_tag.json", "./tests/files/cart_no_tag.blob"),
        ]
    )
    def test_encode_chunk(self, json_filename: str, blob_filename: str):
        """
        Encode a cart chunk.
        """

        # Arrange

        with open(json_filename, "r") as json_file:
            fields = json.load(json_file)

        timers = []

        for timer_parts in fields["timers"]:
            timers.append(CartTimer(timer_parts["name"], timer_parts["time"]))

        chunk = CartChunk(
            fields["version"],
            fields["title"],
            fields["artist"],
            fields["cut_id"],
            fields["client_id"],
            fields["category"],
            fields["classification"],
            fields["out_cue"],
            datetime.strptime(fields["start_date"], CartChunk.FORMAT_DATE_TIME),
            datetime.strptime(fields["end_date"], CartChunk.FORMAT_DATE_TIME),
            fields["producer_app"],
            fields["producer_app_version"],
            fields["user_defined"],
            int(fields["ref_0db"]),
            timers,
            fields["url"],
            fields["tag_text"],
        )

        with open(blob_filename, "rb") as blob_file:
            expected_blob = blob_file.read()

        # Act

        blob = chunk.to_bytes()

        # Assert

        self.assertEqual(len(blob), len(expected_blob))
        self.assertEqual(blob, expected_blob)

    def test_cart_bad_length(self):
        """
        An error is raised when an invalid cart length is supplied.
        """

        #  Arrange

        with open("./tests/files/cart_bad_length.blob", "rb") as cart_file:

            # Act

            with self.assertRaises(InvalidHeaderException) as context:
                CartChunk.from_file(cart_file, 0)

                # Assert

                self.assertIn(
                    "Cart chunk is not long enough. Must be a minimum of 2048 bytes",
                    context.exception,
                )

    @parameterized.expand(
        [
            ("ERR0", 0, "ERR is not a valid timer prefix"),
            ("MRKs", 0, "MRK timers cannot have start or end suffixes"),
            ("MRKe", 0, "MRK timers cannot have start or end suffixes"),
            ("AUD1", 0, "AUD timers cannot be enumerated"),
        ]
    )
    def test_invalid_timer_names(self, name: str, time: int, expected_error: str):
        """
        Invalid timer names raise an error.
        """

        # Arrange

        # Act

        with self.assertRaises(InvalidTimerException) as context:
            CartTimer(name, time)

            # Assert

            self.assertIn(expected_error, context.exception)
