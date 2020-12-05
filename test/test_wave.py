from exceptions import InvalidHeaderException, InvalidWaveException
from chunks import CartChunk, CartTimer, DataChunk, FormatChunk, RiffChunk, WaveFormat
from datetime import datetime
import numpy as np
from parameterized import parameterized
from typing import List
from unittest import TestCase


class TestWaveChunk(TestCase):
    @parameterized.expand(
        [
            ("./test/files/valid_no_markers.wav", [b"fmt ", b"data"]),
        ]
    )
    def test_read_valid_wave(self, file_name: str, expected_chunks: List[str]):
        """
        Read valid wave files.
        """

        # Arrange

        with open(file_name, "rb") as file:

            # Act

            chunk = RiffChunk.from_file(file)

            # Assert

            self.assertIsNotNone(chunk)
            self.assertEqual(chunk.get_name, b"WAVE")
            self.assertIsNotNone(chunk.sub_chunks)
            self.assertEqual(len(chunk.sub_chunks), len(expected_chunks))

            for expected_chunk in expected_chunks:
                self.assertIn(expected_chunk, chunk.sub_chunks)

    def test_encode_wave_with_cart(self):
        """
        A WAVE file with a cart chunk can be encoded.
        """

        # Arrange

        chunks = {}

        chunks[RiffChunk.CHUNK_FORMAT] = FormatChunk(
            WaveFormat.PCM, False, 2, 44100, 16
        )

        timers = [
            CartTimer("INTs", 0),
            CartTimer("INTe", 41373),
            CartTimer("SEG ", 108118),
        ]

        chunks[RiffChunk.CHUNK_CART] = CartChunk(
            "0101",
            "Test Cart Title",
            "Test Cart Artist",
            "TESTCART01",
            "Someone",
            "DEMO",
            "Demo Audio",
            "Radio!",
            datetime(1900, 1, 1, 0, 0),
            datetime(2099, 12, 31, 23, 59, 59),
            "Hand Crafted",
            "MK1 Eyeball",
            "Some stuff goes in here....",
            32768,
            timers,
            "http://www.example.com/",
            "Load of text goes in here.\r\n",
        )

        with open("./test/files/valid_no_markers.wav", "rb") as in_file:
            samples = np.memmap(
                in_file, dtype=np.dtype("<i2"), mode="c", shape=(111020, 2), offset=44
            )

        chunks[RiffChunk.CHUNK_DATA] = DataChunk(samples)

        riff = RiffChunk(chunks)

        with open("./test/files/valid_with_markers.wav", "rb") as expected_file:
            expected_blob = expected_file.read()

        #  Act

        blob = riff.to_bytes()

        # Assert

        self.assertIsNotNone(blob)
        self.assertEqual(blob, expected_blob)

    def test_encode_wave_no_cart(self):
        """
        A WAVE file with no cart chunk can be encoded.
        """

        # Arrange

        chunks = {}

        chunks[RiffChunk.CHUNK_FORMAT] = FormatChunk(
            WaveFormat.PCM, False, 2, 44100, 16
        )

        with open("./test/files/valid_no_markers.wav", "rb") as in_file:
            samples = np.memmap(
                in_file, dtype=np.dtype("<i2"), mode="c", shape=(111020, 2), offset=44
            )

        chunks[RiffChunk.CHUNK_DATA] = DataChunk(samples)

        riff = RiffChunk(chunks)

        with open("./test/files/valid_no_markers.wav", "rb") as expected_file:
            expected_blob = expected_file.read()

        #  Act

        blob = riff.to_bytes()

        # Assert

        self.assertIsNotNone(blob)
        self.assertEqual(blob, expected_blob)

    def test_riff_bad_header(self):
        """
        Raise an exception if the header does not start with RIFF
        """

        # Arrange

        with open("./test/files/cart_long.blob", "rb") as file:

            #  Act

            with self.assertRaises(InvalidHeaderException) as context:
                RiffChunk.from_file(file)

                # Assert

                self.assertIn("WAVE files must have a RIFF header", context.exception)

    def test_riff_bad_length(self):
        """
        Raise an exception if the header does not start a valid length
        """

        # Arrange

        with open("./test/files/riff_bad_length.blob", "rb") as file:

            #  Act

            with self.assertRaises(InvalidHeaderException) as context:
                RiffChunk.from_file(file)

                # Assert

                self.assertIn(
                    "WAVE files must have a length greater than zero", context.exception
                )

    def test_riff_bad_type(self):
        """
        Raise an exception if the RIFF file is not WAVE audio
        """

        # Arrange

        with open("./test/files/riff_bad_type.blob", "rb") as file:

            #  Act

            with self.assertRaises(InvalidHeaderException) as context:
                RiffChunk.from_file(file)

                # Assert

                self.assertIn(
                    "This library only supports WAVE files", context.exception
                )

    def test_riff_wrong_order(self):
        """
        Raise an exception if the sub chunks are in the wrong order
        """

        # Arrange

        with open("./test/files/invalid_wrong_order.wav", "rb") as file:

            #  Act

            with self.assertRaises(InvalidWaveException) as context:
                RiffChunk.from_file(file)

                # Assert

                self.assertIn(
                    "A format chunk must be read before a data chunk!",
                    context.exception,
                )

    def test_encode_no_format(self):
        """
        Raises an exception if no format chunk is supplied.
        """

        # Arrange

        chunk = RiffChunk({})

        # Act

        with self.assertRaises(InvalidWaveException) as context:
            chunk.to_bytes()

            # Assert

            self.assertIn(
                "Valid wave files must have a format chunk", context.exception
            )

    def test_encode_no_data(self):
        """
        Raises an exception if no data chunk is supplied.
        """

        # Arrange

        chunk = RiffChunk({b"fmt ": FormatChunk(WaveFormat.PCM, False, 2, 48000, 16)})

        # Act

        with self.assertRaises(InvalidWaveException) as context:
            chunk.to_bytes()

            # Assert

            self.assertIn(
                "Valid wave files must have a format chunk", context.exception
            )
