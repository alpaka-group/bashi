"""Application argument handling for bashi based validator apps."""

import argparse

from typing import NamedTuple, List, Dict, Sequence, Any, Tuple
from typeguard import typechecked
import packaging.version
from bashi.types import Parameter, ParameterValue
from bashi.globals import (
    HOST_COMPILER,
    DEVICE_COMPILER,
    COMPILERS,
    BACKENDS,
    CXX_STANDARD,
    UBUNTU,
    CMAKE,
    BOOST,
    ALPAKA_ACC_GPU_CUDA_ENABLE,
    ON,
    OFF,
)
from bashi.utils import PARAMETER_SHORT_NAME
from bashi.versions import VERSIONS
from bashiValidate.utils import exit_error

ArgumentAlias = NamedTuple("ArgumentAlias", [("alias", List[str]), ("parameter", Parameter)])


@typechecked
class VersionCheck(argparse.Action):
    """Verify that version can be parsed to package.version.Version.

    Handles special values "ON" and "OFF", which will be parsed to 1.0.0 and 0.0.0
    """

    # check if argument has valid version shape
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
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
            parsed_version = packaging.version.parse(version)
            setattr(namespace, self.dest, parsed_version)
            if option_string:
                parser.param_order.append(option_string)  # type: ignore
        except packaging.version.InvalidVersion:
            exit_error(f"Could not parse version of argument {option_string}: {version}")


@typechecked
class CompilerVersionCheck(argparse.Action):
    """Tries to parse compiler versions string of the shape "name@version" to a parameter-value."""

    # check if argument has valid version shape
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
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
            parsed_version = packaging.version.parse(version)
            setattr(namespace, self.dest, ParameterValue(name, parsed_version))
            if option_string:
                parser.param_order.append(option_string)  # type: ignore
        except packaging.version.InvalidVersion:
            exit_error(f"Could not parse version number of {name}: {version}")


@typechecked
class AliasParser(argparse.Action):
    """Takes a string and maps to a package.version.Version.

    Internal, bashi filters works only with version numbers. Therefore if value-name should be a
    string, it needs to be mapped to a fake version number. This parser allows to use the string
    representation instead the version number as CLI argument.

    e.g. --build_type=Release instead --build_type=0
    """

    # check if argument has valid version shape
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ):
        if not values:
            exit_error(f"argument {option_string}: expected one argument")

        striped_option: str = str(option_string).lstrip("-")
        # if striped_option is in the argument_aliases dict, it is a short name
        # in the case use the regular name instead
        if striped_option in parser.argument_aliases:  # type: ignore
            striped_option = parser.argument_aliases[striped_option]  # type: ignore

        argument_value_aliases: Dict[str, Dict[str, packaging.version.Version]] = (
            parser.argument_value_aliases  # type: ignore
        )

        if striped_option not in argument_value_aliases:
            exit_error(f"no argument value alias for argument {option_string}")

        if values not in argument_value_aliases[striped_option]:
            exit_error(f"no value alias for argument value {values}")

        setattr(namespace, self.dest, argument_value_aliases[striped_option][str(values)])
        if option_string:
            parser.param_order.append(option_string)  # type: ignore


# pylint: disable=too-many-locals
@typechecked
def get_validator_args() -> Tuple[argparse.ArgumentParser, Dict[str, ArgumentAlias], List[str]]:
    """Set up command line arguments.

    Returns:
        argparse.ArgumentParser: An argparse ArgumentParser, which can be extended with more
            arguments.
        Dict[str, ArgumentAlias]: A dict of command line command aliases for an parameter. The
            parameter names are the keys of the dict.
        List[str]: The ordering of the parameters, how the user type it in. The ordering is
            important to trigger a specific rule first.
    """
    args_alias: Dict[str, ArgumentAlias] = {}
    parser = argparse.ArgumentParser(description="Check if combination of parameters is valid.")
    param_order: List[str] = []
    setattr(parser, "param_order", param_order)
    # store for which argument which value should be mapped to packaging.version.Version
    argument_value_aliases: Dict[str, Dict[str, packaging.version.Version]] = {}
    setattr(parser, "argument_value_aliases", argument_value_aliases)
    # map shortname of an argument to the regular argument
    argument_aliases: Dict[str, str] = {}
    setattr(parser, "argument_aliases", argument_aliases)

    def add_param_alias(argument: str, args_alias: Dict[str, ArgumentAlias]) -> List[str]:
        """Returns the argument name and also an alias, if it is defined in the PARAMETER_SHORT_NAME

        Args:
            argument (str): Name of the argument without '--' prefix
            args_alias (Dict[str, ArgumentAlias]): Stores the alias and it's parameter for an
                argument

        Raises:
            ValueError: If parameter is unknown

        Returns:
            List[str]: List of arguments for argparse
        """
        argument_alias = [f"--{argument}"]
        modified_arg = argument
        if argument == "host-compiler":
            modified_arg = HOST_COMPILER

        if argument == "device-compiler":
            modified_arg = DEVICE_COMPILER

        if argument == "cxx":
            modified_arg = CXX_STANDARD

        if not modified_arg in (
            HOST_COMPILER,
            DEVICE_COMPILER,
            *BACKENDS,
            UBUNTU,
            CMAKE,
            BOOST,
            CXX_STANDARD,
        ):
            raise ValueError(f"{modified_arg} is not a know Parameter")

        if modified_arg in PARAMETER_SHORT_NAME:
            argument_alias.append(f"--{PARAMETER_SHORT_NAME[modified_arg]}")

        # argparse also replace the '-' with the '_' if it stores the argument
        args_alias[argument.replace("-", "_")] = ArgumentAlias(argument_alias, modified_arg)

        return argument_alias

    compiler_arg_group = parser.add_argument_group("Compiler")

    compiler_arg_group.add_argument(
        *add_param_alias("host-compiler", args_alias),
        type=str,
        action=CompilerVersionCheck,
        help="Define host compiler. Shape needs to be name@version. " "For example gcc@10",
    )

    compiler_arg_group.add_argument(
        *add_param_alias("device-compiler", args_alias),
        type=str,
        action=CompilerVersionCheck,
        help="Define device compiler. Shape needs to be name@version. " "For example nvcc@11.3",
    )

    backends_arg_group = parser.add_argument_group("Backends")

    for backend in BACKENDS:
        if backend != ALPAKA_ACC_GPU_CUDA_ENABLE:
            backends_arg_group.add_argument(
                *add_param_alias(backend, args_alias),
                type=str,
                action=VersionCheck,
                choices=["ON", "OFF"],
                help=f"Set backend {backend} as enabled or disabled.",
            )
        else:
            backends_arg_group.add_argument(
                *add_param_alias(backend, args_alias),
                type=str,
                action=VersionCheck,
                help=f"Set backend {backend} to disabled (OFF) or a specific CUDA SDK version.",
            )

    other_sw_arg_group = parser.add_argument_group("Other Combination Parameter")

    for argument, help_text in (
        ("ubuntu", "Ubuntu version."),
        ("cmake", "Set CMake version."),
        ("boost", "Set Boost version."),
        ("cxx", "C++ version."),
    ):
        other_sw_arg_group.add_argument(
            *add_param_alias(argument, args_alias),
            type=str,
            action=VersionCheck,
            help=help_text,
        )

    sw_versions_arg_group = parser.add_argument_group("Software Versions")

    for sw_name, sw_version in VERSIONS.items():
        parsed_sw_versions = [str(ver) for ver in sw_version]
        sw_versions_arg_group.add_argument(
            f"--ver-{sw_name}",
            type=str,
            nargs="*",
            default=parsed_sw_versions,
            help=f"Set input version for {sw_name}. Default: {parsed_sw_versions}.",
        )

    return parser, args_alias, param_order
