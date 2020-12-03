from chunks import CartChunk, CartTimer
from datetime import datetime
from exceptions import InvalidHeaderException, InvalidTimerException
import json
from parameterized import parameterized
from typing import List, Tuple
from unittest import TestCase


class TestCartChunk(TestCase):
    @parameterized.expand(
        [
            (
                "./test/files/cc_0101.wav",
                746,
                "0101",
                "Cart Chunk: the traffic data file format for the Radio Industry",
                "Jay Rose, dplay.com",
                "DEMO-0101",
                "CartChunk.org",
                "DEMO",
                "Demo and sample files",
                "the Radio Industry",
                datetime(1900, 1, 1, 0, 0),
                datetime(2099, 12, 31, 23, 59, 59),
                "AUDICY",
                "3.10/623",
                "Demo ID showing basic 'cart' chunk attributes",
                32768,
                [("MRK ", 112000), ("SEC1", 152533), ("EOD", 201024)],
                "http://www.cartchunk.org",
                "The radio traffic data, or 'cart' format utilizes a widely\r\nused standard audio file format (wave and broadcast wave file).\r\nIt incorporates the common broadcast-specific cart labeling\r\ninformation into a specialized chunk within the file itself.\r\nAs a result, the burden of linking multiple systems is reduced\r\nto producer applications writing a single file, and the consumer\r\napplications reading it. The destination application can extract\r\ninformation and insert it into the native database application\r\nas needed.\r\n",
            ),
            (
                "./test/files/cart_no_tag.blob",
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
                "./test/files/valid_no_markers.wav",
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
            ("./test/files/cart_cc_101.json", "./test/files/cart_cc_101.blob"),
            ("./test/files/cart_long.json", "./test/files/cart_long.blob"),
            ("./test/files/cart_no_tag.json", "./test/files/cart_no_tag.blob"),
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

        with open("./test/files/cart_bad_length.blob", "rb") as cart_file:

            # Act

            with self.assertRaises(InvalidHeaderException) as context:
                CartChunk.from_file(cart_file, 0)

                # Assert

                self.assertIn(
                    "Cart chunk is not long enough. Must be a minimum of 2048 bytes"
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
