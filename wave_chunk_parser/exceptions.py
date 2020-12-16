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


class InvalidWaveException(Exception):
    """
    The wave file is not valid.
    """
