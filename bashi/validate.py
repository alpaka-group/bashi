#!/usr/bin/env python3

"""Command line tool to check filter rules."""

import argparse
from argparse import ArgumentParser, Namespace

from typing import Sequence, Any, Callable, Optional, IO
from collections import OrderedDict
import io
import sys
from typeguard import typechecked
import packaging.version as pkv

from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValue, ParameterValueTuple
from bashi.versions import is_supported_version
import bashi.filter_compiler
import bashi.filter_backend
import bashi.filter_software_dependency


@typechecked
def cs(text: str, color: str) -> str:
    """Prints colored text to the command line. The text printed after the function call has the
    default color of the command line.

    Args:
        text (str): text to be colored
        color (str): Name of the color. If color is unknown or empty use default color of the
            command line.

    Returns:
        str: text with bash control symbols for coloring
    """

    output = ""
    if color == "Red":
        output += "\033[0;31m"
    elif color == "Green":
        output += "\033[0;32m"
    elif color == "Yellow":
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
    print(cs("ERROR: " + text, "Red"))
    sys.exit(1)


@typechecked
class VersionCheck(argparse.Action):
    """Verify that version can be parsed to package.version.Version.

    Handles special values "ON" and "OFF", which will be parsed to 1.0.0 and 0.0.0
    """

    # check if argument has valid version shape
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ):
        if not values:
            exit_error(f"argument {option_string}: expected one argument")

        version: str = str(values)

        if version == "OFF":
            version = OFF

        if version == "ON":
            version = ON

        # use parse() function to validate that the version has a valid shape
        try:
            parsed_version = pkv.parse(version)
            setattr(namespace, self.dest, parsed_version)
        except packaging.version.InvalidVersion:
            exit_error(f"Could not parse version of argument {option_string}: {version}")


@typechecked
class CompilerVersionCheck(argparse.Action):
    """Tries to parse compiler versions string of the shape "name@version" to a parameter-value."""

    # check if argument has valid version shape
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        if not values:
            exit_error(f"argument {option_string}: expected one argument")

        parameter_value_str: str = str(values)

        if "@" not in parameter_value_str:
            exit_error(f"@ is missing in {option_string}={parameter_value_str}")
            return

        splitted_name_version = parameter_value_str.split("@", 1)
        name = splitted_name_version[0]
        version = splitted_name_version[1]

        if name not in COMPILERS:
            exit_error(f"Unknown compiler: {name}\nKnown compilers: {COMPILERS}")

        # use parse() function to validate that the version has a valid shape
        try:
            parsed_version = pkv.parse(version)
            setattr(namespace, self.dest, ParameterValue(name, parsed_version))
        except packaging.version.InvalidVersion:
            exit_error(f"Could not parse version number of {name}: {version}")


def get_args() -> Namespace:
    """Set up command line arguments and parsed it.

    Returns:
        Namespace: The parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Check if combination of parameters is valid.")

    parser.add_argument(
        "--host-compiler",
        type=str,
        action=CompilerVersionCheck,
        help="Define host compiler. Shape needs to be name@version. " "For example gcc@10",
    )

    parser.add_argument(
        "--device-compiler",
        type=str,
        action=CompilerVersionCheck,
        help="Define device compiler. Shape needs to be name@version. " "For example nvcc@11.3",
    )

    for backend in BACKENDS:
        if backend != ALPAKA_ACC_GPU_CUDA_ENABLE:
            parser.add_argument(
                "--" + backend,
                type=str,
                action=VersionCheck,
                choices=["ON", "OFF"],
                help=f"Set backend {backend} as enabled or disabled.",
            )
        else:
            parser.add_argument(
                "--" + backend,
                type=str,
                action=VersionCheck,
                help=f"Set backend {backend} to disabled (OFF) or a specific CUDA SDK version.",
            )

    parser.add_argument("--ubuntu", type=str, action=VersionCheck, help="Ubuntu version.")

    parser.add_argument("--cmake", type=str, action=VersionCheck, help="Set CMake version.")

    parser.add_argument("--boost", type=str, action=VersionCheck, help="Set Boost version.")

    parser.add_argument("--cxx", type=str, action=VersionCheck, help="C++ version.")

    return parser.parse_args()


@typechecked
def check_single_filter(
    filter_func: Callable[[ParameterValueTuple, Optional[IO[str]]], bool],
    row: ParameterValueTuple,
) -> bool:
    """Check if row passes a filter function.

    Args:
        filter_func (Callable[[ParameterValueTuple, Optional[IO[str]]], bool]): The filter function
        row (ParameterValueTuple): row with parameter-value-tuples
        required_parameter (List[str]): list of parameters, which will be used in the filter rule

    Returns:
        bool: True if the row passes the filter.
    """
    # get name of the filter for command line output.
    filter_name: str = filter_func.__name__

    # Each filter function has also type checked version ending with _typed.
    # Remove _typed for better output.
    if filter_name.endswith("_typechecked"):
        filter_name = filter_name[: -len("_typechecked")]

    msg = io.StringIO()

    if filter_func(row, msg):
        print(cs(f"{filter_name}() returns True", "Green"))
        return True

    print(cs(f"{filter_name}() returns False", "Red"))
    if msg.getvalue() != "":
        print("  " + msg.getvalue())
    return False


@typechecked
def check_filter_chain(row: ParameterValueTuple) -> bool:
    """Test a row with the bashi default filter chain.

    Args:
        row (ParameterValueTuple): row to test

    Returns:
        bool: True if row passes all filters
    """
    all_true = 0
    all_true += int(
        check_single_filter(
            bashi.filter_compiler.compiler_filter,
            row,
        )
    )
    all_true += int(
        check_single_filter(
            bashi.filter_backend.backend_filter_typechecked,
            row,
        )
    )
    all_true += int(
        check_single_filter(
            bashi.filter_software_dependency.software_dependency_filter_typechecked,
            row,
        )
    )

    # each filter add a one, if it was successful
    return all_true == 3


def main() -> None:
    """Entry point for the application."""
    args = get_args()

    row: ParameterValueTuple = OrderedDict()

    for param, arg in [
        (HOST_COMPILER, "host_compiler"),
        (DEVICE_COMPILER, "device_compiler"),
    ]:
        if hasattr(args, arg) and getattr(args, arg) is not None:
            row[param] = getattr(args, arg)

    for backend in BACKENDS:
        if hasattr(args, backend) and getattr(args, backend) is not None:
            row[backend] = ParameterValue(backend, getattr(args, backend))

    for param, arg in [
        (UBUNTU, "ubuntu"),
        (CMAKE, "cmake"),
        (BOOST, "boost"),
        (CXX_STANDARD, "cxx"),
    ]:
        if hasattr(args, arg) and getattr(args, arg) is not None:
            row[param] = ParameterValue(param, getattr(args, arg))

    for val_name, val_version in row.values():
        if not is_supported_version(val_name, val_version):
            print(
                cs(
                    f"WARNING: {val_name} {val_version} is not officially supported.",
                    "Yellow",
                )
            )

    sys.exit(int(not check_filter_chain(row)))


if __name__ == "__main__":
    main()
