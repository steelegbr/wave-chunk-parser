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
