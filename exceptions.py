class InvalidHeaderException(Exception):
    """
    Indicates the header is invalid.
    """

    pass


class ExportExtendedFormatException(Exception):
    """
    Indicates that a request to export an extended format header was made. We don't support that.
    """

    pass


class InvalidTimerException(Exception):
    """
    Indicates a cart timer is invalid.
    """

    pass
