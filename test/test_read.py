from chunks import FormatChunk, WaveFormat
from parameterized import parameterized
from unittest import TestCase


class TestReadFiles(TestCase):
    @parameterized.expand(
        [
            (
                "./test/files/valid_no_markers.wav",
                12,
                WaveFormat.PCM,
                2,
                44100,
                176400,
                4,
                16,
            )
        ]
    )
    def test_read_valid_format_chunk(
        self,
        file_name: str,
        chunk_offset: int,
        expected_format: WaveFormat,
        expected_channels: int,
        expected_sample_rate: int,
        expected_byte_rate: int,
        expected_block_align: int,
        expected_bits_per_sample: int,
    ):
        """
        Valid format chunks can be read from a file.
        """

        # Arrange

        with open(file_name, "rb") as file:

            # Act

            chunk: FormatChunk = FormatChunk.from_file(file, chunk_offset)

            # Assert

            self.assertIsNotNone(chunk)
            self.assertEqual(chunk.format, expected_format)
            self.assertEqual(chunk.channels, expected_channels)
            self.assertEqual(chunk.sample_rate, expected_sample_rate)
            self.assertEqual(chunk.byte_rate, expected_byte_rate)
            self.assertEqual(chunk.block_align, expected_block_align)
            self.assertEqual(chunk.bits_per_sample, expected_bits_per_sample)
