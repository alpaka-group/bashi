"""Several functions to print bashi data structures."""

from typing import Dict
from bashi.types import (
    Parameter,
    ParameterValue,
    ParameterValueSingle,
    ParameterValueTuple,
    ValueName,
    ValueVersion,
)
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


# short names for parameter
PARAMETER_SHORT_NAME: dict[Parameter, str] = {
    HOST_COMPILER: "host",
    DEVICE_COMPILER: "device",
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: "bOpenMP2thread",
    ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE: "bOpenMP2block",
    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: "bSeq",
    ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE: "bThreads",
    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE: "bTBB",
    ALPAKA_ACC_GPU_CUDA_ENABLE: "bCUDA",
    ALPAKA_ACC_GPU_HIP_ENABLE: "bHIP",
    ALPAKA_ACC_ONEAPI_CPU_ENABLE: "bSYCLcpu",
    ALPAKA_ACC_ONEAPI_GPU_ENABLE: "bSYCLgpu",
    ALPAKA_ACC_ONEAPI_FPGA_ENABLE: "bSYCLfpga",
    CXX_STANDARD: "c++",
}


print_row_nice_parameter_alias: Dict[Parameter, str] = PARAMETER_SHORT_NAME.copy()
print_row_nice_version_aliases: Dict[ValueName, Dict[ValueVersion, str]] = {}


# do not cover code, because the function is only used for debugging
def add_print_row_nice_parameter_alias(parameter_name: Parameter, alias: str):  # pragma: no cover
    """Add an alias for an parameter, which will be displayed if print_row_nice() is called.

    Args:
        parameter_name (Parameter): parameter
        alias (str): alias
    """
    print_row_nice_parameter_alias[parameter_name] = alias


def add_print_row_nice_version_alias(
    value_name: ValueName, versions_aliases: Dict[ValueVersion, str]
):  # pragma: no cover
    """Add an aliases for the version of parameter-value, which will be displayed if
    print_row_nice() is called.

    Args:
        value_name (ValueName): parameter-name
        versions_aliases (Dict[ValueVersion, str]): text which is display instead the value-version
    """
    print_row_nice_version_aliases[value_name] = versions_aliases


def get_str_parameter_value(parameter_value: ParameterValue) -> str:  # pragma: no cover
    """Returns a parameter-value as string in a short and nice way."""
    nice_version: dict[packaging.version.Version, str] = {
        ON_VER: "ON",
        OFF_VER: "OFF",
    }

    if (
        parameter_value.name in print_row_nice_version_aliases
        and parameter_value.version in print_row_nice_version_aliases[parameter_value.name]
    ):
        return f"{print_row_nice_version_aliases[parameter_value.name][parameter_value.version]}"

    return f"{nice_version.get(parameter_value.version, str(parameter_value.version))}"


def get_str_parameter_tuple(
    parameter: Parameter, parameter_value: ParameterValue
) -> str:  # pragma: no cover
    """Returns a parameter and parameter-value as string in a short and nice way."""
    nice_version: dict[packaging.version.Version, str] = {
        ON_VER: "ON",
        OFF_VER: "OFF",
    }

    s = f"{print_row_nice_parameter_alias.get(parameter, parameter)}="

    if parameter in [HOST_COMPILER, DEVICE_COMPILER]:
        s += (
            f"{print_row_nice_parameter_alias.get(parameter_value.name, parameter_value.name)}@"
            f"{nice_version.get(parameter_value.version, str(parameter_value.version))}"
        )
    else:
        s += get_str_parameter_value(parameter_value)
    return s


def get_str_parameter_value_single(
    parameter_value_single: ParameterValueSingle,
) -> str:  # pragma: no cover
    """Returns a parameter-value-single as string in a short and nice way."""
    return get_str_parameter_tuple(
        parameter_value_single.parameter, parameter_value_single.parameterValue
    )


def get_str_row_nice(
    row: ParameterValueTuple, init: str = "", bashi_validate: bool = False
) -> str:  # pragma: no cover
    """Returns a parameter-value-tuple as string in a short and nice way.

    Args:
        row (ParameterValueTuple): row with parameter-value-tuple
        init (str, optional): Prefix of the output string. Defaults to "".
        bashi_validate (bool): If it is set to True, the row is printed in a form that can be passed
            directly as arguments to bashi-validate. Defaults to False.
    Return:
        str: string representation of a parameter-value-tuple
    """
    s = init

    for param, val in row.items():
        parameter_prefix = "" if not bashi_validate else "--"
        s += f"{parameter_prefix}{get_str_parameter_tuple(param, val)} "

    return s


def print_row_nice(
    row: ParameterValueTuple, init: str = "", bashi_validate: bool = False
):  # pragma: no cover
    """Prints a parameter-value-tuple in a short and nice way.

    Args:
        row (ParameterValueTuple): row with parameter-value-tuple
        init (str, optional): Prefix of the output string. Defaults to "".
        bashi_validate (bool): If it is set to True, the row is printed in a form that can be passed
            directly as arguments to bashi-validate. Defaults to False.
    """
    print(get_str_row_nice(row, init, bashi_validate))
