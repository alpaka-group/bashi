"""Utility function for bashi based validator applications."""

import sys
from typeguard import typechecked
import termcolor


@typechecked
def exit_error(text: str):
    """Prints error message and exits application with error code 1.

    Args:
        text (str): Error message.
    """
    print(termcolor.colored("ERROR: " + text, "red"))
    sys.exit(1)
