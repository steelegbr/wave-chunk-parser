from utils import seek_and_read
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
