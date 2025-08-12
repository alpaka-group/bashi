"""Utility function for bashi based validator applications."""

import sys
from enum import Enum
from typeguard import typechecked


class Color(Enum):
    """Color for the cs() function."""

    RED = 1
    GREEN = 2
    YELLOW = 3


@typechecked
def cs(text: str, color: Color) -> str:
    """Prints colored text to the command line. The text printed after the function call has the
    default color of the command line.

    Args:
        text (str): Text to be colored
        color (Color): Text color

    Returns:
        str: text with bash control symbols for coloring
    """

    output = ""
    if color == Color.RED:
        output += "\033[0;31m"
    elif color == Color.GREEN:
        output += "\033[0;32m"
    elif color == Color.YELLOW:
        output += "\033[1;33m"
    else:
        return text

    return output + text + "\033[0m"


@typechecked
def exit_error(text: str):
    """Prints error message and exits application with error code 1.

    Args:
        text (str): Error message.
    """
    print(cs("ERROR: " + text, Color.RED))
    sys.exit(1)
