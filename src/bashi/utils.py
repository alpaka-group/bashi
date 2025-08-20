"""Different helper functions for bashi"""

import sys
from typing import IO, Dict, List, Optional, Union, Callable

from packaging.specifiers import SpecifierSet
from typeguard import typechecked

from bashi.types import (
    CombinationList,
    Parameter,
    ParameterValue,
    ParameterValueMatrix,
    ParameterValuePair,
    ParameterValueSingle,
    ParameterValueTuple,
    ValueName,
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


# pylint: disable=too-many-positional-arguments
@typechecked
def create_parameter_value_pair(  # pylint: disable=too-many-arguments
    parameter1: str,
    value_name1: str,
    value_version1: Union[int, float, str, packaging.version.Version],
    parameter2: str,
    value_name2: str,
    value_version2: Union[int, float, str, packaging.version.Version],
) -> ParameterValuePair:
    """Create parameter-value-pair from the given arguments. Parse parameter-versions if required.

    Args:
        parameter1 (str): name of the first parameter
        value_name1 (str): name of the first value-name
        value_version1 (Union[int, float, str, packaging.version.Version]): version of first
            value-version
        parameter2 (str): name of the second parameter
        value_name2 (str): name of the second value-name
        value_version2 (Union[int, float, str, packaging.version.Version]): version of the second
            value-version

    Returns:
        ParameterValuePair: parameter-value-pair
    """
    if isinstance(value_version1, packaging.version.Version):
        parsed_value_version1: packaging.version.Version = value_version1
    else:
        parsed_value_version1: packaging.version.Version = packaging.version.parse(  # type: ignore
            str(value_version1)
        )

    if isinstance(value_version2, packaging.version.Version):
        parsed_value_version2: packaging.version.Version = value_version2
    else:
        parsed_value_version2: packaging.version.Version = packaging.version.parse(  # type: ignore
            str(value_version2)
        )

    return ParameterValuePair(
        ParameterValueSingle(parameter1, ParameterValue(value_name1, parsed_value_version1)),
        ParameterValueSingle(parameter2, ParameterValue(value_name2, parsed_value_version2)),
    )


@typechecked
def get_expected_parameter_value_pairs(
    parameter_matrix: ParameterValueMatrix,
) -> List[ParameterValuePair]:
    """Takes parameter-value-matrix and creates a list of all expected parameter-values-pairs.
    The pair-wise generator guaranties, that each pair of two parameter-values exist in at least one
    combination if no filter rules exist. Therefore the generated the generated list can be used
    to verify the output of the pair-wise generator.

    Args:
        parameter_matrix (ParameterValueMatrix): matrix of parameter values

    Returns:
        List[ParameterValuePair]: list of all possible parameter-value-pairs
    """
    expected_pairs: List[ParameterValuePair] = []

    number_of_keys: int = len(parameter_matrix.keys())
    param_map: Dict[int, str] = {}
    for index, param in enumerate(parameter_matrix.keys()):
        param_map[index] = param

    for v1_index in range(number_of_keys):
        for v2_index in range(v1_index + 1, number_of_keys):
            _loop_over_parameter_values(
                parameter_matrix,
                expected_pairs,
                param_map[v1_index],
                param_map[v2_index],
            )

    return expected_pairs


@typechecked
def _loop_over_parameter_values(
    parameters: ParameterValueMatrix,
    expected_pairs: List[ParameterValuePair],
    v1_parameter: Parameter,
    v2_parameter: Parameter,
):
    """Creates all parameter-value-pairs for two give parameters.

    Args:
        parameters (ParameterValueMatrix): The complete parameter-value-matrix
        expected_pairs (List[ParameterValuePair]): Add the generated parameter-values-pairs to the
            list.
        v1_parameter (Parameter): the first parameter
        v2_parameter (Parameter): the second parameter
    """
    for v1_name, v1_version in parameters[v1_parameter]:
        for v2_name, v2_version in parameters[v2_parameter]:
            expected_pairs.append(
                create_parameter_value_pair(
                    v1_parameter, v1_name, v1_version, v2_parameter, v2_name, v2_version
                )
            )


@typechecked
def bi_filter(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    filter_function: Callable[[ParameterValuePair], bool],
):
    """Filtering of parameter-value-pairs according to the specified filter function and put the
    filtered entries in the list of removed parameter-value-pairs.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List to be filtered
        removed_parameter_value_pairs (List[ParameterValuePair]): List into which the filtered
            elements are inserted
        filter_function (Callable[[ParameterValuePair], bool]): Filter function. Returns true if the
            element is to remain in parameter_value_pairs.
    """
    tmp_parameter_value_pairs: List[ParameterValuePair] = []

    for param_val_pair in parameter_value_pairs:
        if filter_function(param_val_pair):
            tmp_parameter_value_pairs.append(param_val_pair)
        else:
            removed_parameter_value_pairs.append(param_val_pair)

    parameter_value_pairs[:] = tmp_parameter_value_pairs


def get_nice_paremter_value_pair_str(param_val_pair: ParameterValuePair) -> str:
    """Take Parameter value pair and return a nice string representation.

    Args:
        param_val_pair (ParameterValuePair): Parameter value pair

    Returns:
        str: string representation
    """
    output = "( "
    for i, param_val_single in enumerate(param_val_pair):
        if param_val_single.parameter in (HOST_COMPILER, DEVICE_COMPILER):
            output += f"{param_val_single.parameter} "
        output += (
            f"{param_val_single.parameterValue.name}"
            f"@{str(param_val_single.parameterValue.version)} "
        )
        if i == 0:
            output += ", "
    output += ")"
    return output


@typechecked
def check_parameter_value_pair_in_combination_list(
    combination_list: CombinationList,
    parameter_value_pairs: List[ParameterValuePair],
    output: IO[str] = sys.stdout,
) -> bool:
    """Check if all given parameter-values-pairs exist at least in on combination.

    Args:
        combination_list (CombinationList): list of given combination
        parameter_value_pairs (List[ParameterValuePair]): list of parameter-value-pair to be search
            for
        output (IO[str], optional): Writes missing parameter-values-pairs to it. Defaults to
            sys.stdout.

    Returns:
        bool: returns True, if all given parameter-values-pairs was found in the combination-list
    """
    missing_expected_param = False

    for ex_param_val_pair in parameter_value_pairs:
        param1, param_val1 = ex_param_val_pair[0]
        param2, param_val2 = ex_param_val_pair[1]
        found = False
        for comb in combination_list:
            # comb contains all parameters, therefore a check is not required
            if comb[param1] == param_val1 and comb[param2] == param_val2:
                found = True
                break

        if not found:
            print(
                f"MISSING in combination list: "
                f"{get_nice_paremter_value_pair_str(ex_param_val_pair)}",
                file=output,
            )
            missing_expected_param = True

    return not missing_expected_param


@typechecked
def check_unexpected_parameter_value_pair_in_combination_list(
    combination_list: CombinationList,
    parameter_value_pairs: List[ParameterValuePair],
    output: IO[str] = sys.stdout,
) -> bool:
    """Check if the given parameter-values-pairs exist in at least in one combination.

    Args:
        combination_list (CombinationList): list of given combination
        parameter_value_pairs (List[ParameterValuePair]): list of parameter-value-pair to be search
            for
        output (IO[str], optional): Writes found parameter-values-pairs to it. Defaults to
            sys.stdout.

    Returns:
        bool: returns True, if no given parameter-values-pairs was found in the combination-list
    """
    found_unexpected_param = False

    for ex_param_val_pair in parameter_value_pairs:
        param1, param_val1 = ex_param_val_pair[0]
        param2, param_val2 = ex_param_val_pair[1]
        for comb in combination_list:
            # comb contains all parameters, therefore a check is not required
            if comb[param1] == param_val1 and comb[param2] == param_val2:
                print(
                    f"FOUND unexpected parameter-value-pair in combination list: "
                    f"{get_nice_paremter_value_pair_str(ex_param_val_pair)}",
                    file=output,
                )
                found_unexpected_param = True
                break

    return not found_unexpected_param


def reason(output: Optional[IO[str]], msg: str):
    """Write the message to output if it is not None. This function is used
    in filter functions to print additional information about filter decisions.

    Args:
        output (Optional[IO[str]]): IO object. For example, can be io.StringIO, sys.stdout or
            sys.stderr
        msg (str): the message
    """
    if output:
        print(
            msg,
            file=output,
            end="",
        )


print_row_nice_parameter_alias: Dict[Parameter, str] = PARAMETER_SHORT_NAME.copy()
print_row_nice_version_aliases: Dict[ValueName, Dict[ValueVersion, str]] = {}


def add_print_row_nice_parameter_alias(parameter_name: Parameter, alias: str):
    """Add an alias for an parameter, which will be displayed if print_row_nice() is called.

    Args:
        parameter_name (Parameter): parameter
        alias (str): alias
    """
    print_row_nice_parameter_alias[parameter_name] = alias


def add_print_row_nice_version_alias(
    value_name: ValueName, versions_aliases: Dict[ValueVersion, str]
):
    """Add an aliases for the version of parameter-value, which will be displayed if
    print_row_nice() is called.

    Args:
        value_name (ValueName): parameter-name
        versions_aliases (Dict[ValueVersion, str]): text which is display instead the value-version
    """
    print_row_nice_version_aliases[value_name] = versions_aliases


# do not cover code, because the function is only used for debugging
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

    nice_version: dict[packaging.version.Version, str] = {
        ON_VER: "ON",
        OFF_VER: "OFF",
    }

    for param, val in row.items():
        parameter_prefix = "" if not bashi_validate else "--"
        if param in [HOST_COMPILER, DEVICE_COMPILER]:
            s += (
                f"{parameter_prefix}{print_row_nice_parameter_alias.get(param, param)}="
                f"{print_row_nice_parameter_alias.get(val.name, val.name)}@"
                f"{nice_version.get(val.version, str(val.version))} "
            )
        else:
            s += f"{parameter_prefix}{print_row_nice_parameter_alias.get(param, param)}="
            if (
                val.name in print_row_nice_version_aliases
                and val.version in print_row_nice_version_aliases[val.name]
            ):
                s += f"{print_row_nice_version_aliases[val.name][val.version]} "
            else:
                s += f"{nice_version.get(val.version, str(val.version))} "
    return s


# do not cover code, because the function is only used for debugging
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


def _create_version_range(
    min_version: Union[int, float, str],
    min_version_inclusive: bool,
    max_version: Union[int, float, str],
    max_version_inclusive: bool,
) -> SpecifierSet:
    """Creates Version SpecifierSet depending on the input.

    Args:
        min_version (Union[int, float, str]): Minimum version of the version range. Must be able
            to be parsed into a `packaging.version.Version`. Use `ANY_VERSION` if the minimum range
            should be open and every check for minimum version should return true.
        min_version_inclusive (bool): If True, the minimum version is within the range and a check
            for minimum version results in True. If False, the check for the minimum version results
            in False.
        max_version (Union[int, float, str]): Maximum version of the version range. Must be able
            to be parsed into a `packaging.version.Version`. Use `ANY_VERSION` if the maximum range
            should be open and every check for maximum version should return true.
        max_version_inclusive (bool): If True, the maximum version is within the range and a check
            for maximum version results in True. If False, the check for the maximum version results
            in False.

    Returns:
        SpecifierSet: A SpecifierSet which can be used to check if version is inside the created
            range.
    """
    # if empty, it matches all versions
    min_range = SpecifierSet()
    max_range = SpecifierSet()

    if min_version != ANY_VERSION:
        # check if valid version number
        packaging.version.parse(str(min_version))
        min_range = SpecifierSet(
            ">=" + str(min_version) if min_version_inclusive else ">" + str(min_version)
        )

    if max_version != ANY_VERSION:
        # check if valid version number
        packaging.version.parse(str(max_version))
        max_range = SpecifierSet(
            "<=" + str(max_version) if max_version_inclusive else "<" + str(max_version)
        )

    return min_range & max_range


# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments
@typechecked
def remove_parameter_value_pairs_ranges(  # pylint: disable=too-many-arguments
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    parameter1: Parameter = ANY_PARAM,
    value_name1: ValueName = ANY_NAME,
    value_min_version1: Union[int, float, str] = ANY_VERSION,
    value_min_version1_inclusive: bool = True,
    value_max_version1: Union[int, float, str] = ANY_VERSION,
    value_max_version1_inclusive: bool = True,
    parameter2: Parameter = ANY_PARAM,
    value_name2: ValueName = ANY_NAME,
    value_min_version2: Union[int, float, str] = ANY_VERSION,
    value_min_version2_inclusive: bool = True,
    value_max_version2: Union[int, float, str] = ANY_VERSION,
    value_max_version2_inclusive: bool = True,
    symmetric: bool = True,
) -> bool:
    """Removes all elements from `parameter_value_pairs` and moves them to
    `removed_parameter_value_pairs` if certain filter requirements are met. The filter properties
    are defined for the first and/or second parameter-value in a parameter-value-pair. All entries
    that meet all requirements are removed from “parameter_value_pairs”.

    Use `remove_parameter_value_pairs()` if you want to remove a single version from the first
    and/or second parameter value. If you want to remove a range of versions, use
    `remove_parameter_value_pairs_ranges()`.

    The default values `ANY_PARAM`, `ANY_NAME` and `ANY_VERSION` match all values of each property,
    which means if each argument is set to default, all elements of `parameter_value_pairs` are
    removed.c

    Parameter and value-name are checked for equality.

    value_min_version and value_max_version allow you to define a version range that is to be
    removed. By default, the version range is open in both directions (minimum and maximum version)
    and can be restricted. If the version range is defined for both parameter values, the pair must
    match both version ranges for it to be removed.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): list where parameter-value-pairs will be
            removed
        removed_parameter_value_pairs (List[ParameterValuePair]): list where removed
            parameter-value-pairs will be stored
        parameter1 (Parameter, optional): Name of the first parameter. Defaults to ANY_PARAM.
        value_name1 (ValueName, optional): Name of the first value-name. Defaults to ANY_NAME.
        value_min_version1 (Union[int, float, str], optional): Minimum version of the version range
            of the first value-version. All versions that are greater than this version are removed.
            Defaults to ANY_VERSION.
        value_min_version1_inclusive (bool, optional): If True, `value_min_version1` is removed.
            Otherwise, all versions greater than `value_min_version1` are removed. Defaults to True.
        value_max_version1 (Union[int, float, str], optional): Maximum version of the version range
            of the first value-version. All versions that are smaller than this version are removed.
            Defaults to ANY_VERSION.
        value_max_version1_inclusive (bool, optional): If True, `value_max_version1` is removed.
            Otherwise, all versions smaller than `value_max_version1` are removed. Defaults to True.
        parameter2 (Parameter, optional): _description_. Defaults to ANY_PARAM.
        value_name2 (ValueName, optional): _description_. Defaults to ANY_NAME.
        value_min_version2 (Union[int, float, str], optional): Minimum version of the version range
            of the second value-version. All versions that are greater than this version are
            removed. Defaults to ANY_VERSION.
        value_min_version2_inclusive (bool, optional): If True, `value_min_version2` is removed.
            Otherwise, all versions greater than `value_min_version2` are removed. Defaults to True.
        value_max_version2 (Union[int, float, str], optional): Maximum version of the version range
            of the second value-version. All versions that are smaller than this version are
            removed. Defaults to ANY_VERSION.
        value_max_version2_inclusive (bool, optional): If True, `value_max_version2` is removed.
            Otherwise, all versions smaller than `value_max_version2` are removed. Defaults to True.
        symmetric (bool, optional): If symmetric is true, it does not matter whether a group of
            parameters, value-name and value-version was found in the first or second
            parameter-value. If false, it is taken into account whether the search criterion was
            found in the first or second parameter value. Defaults to True.

    Returns:
        bool: Return True, if parameter-value-pair was removed.
    """
    filter_list: List[Callable[[ParameterValuePair], bool]] = []
    if parameter1 != ANY_PARAM:
        filter_list.append(lambda param_val: param_val.first.parameter == parameter1)

    if value_name1 != ANY_NAME:
        filter_list.append(lambda param_val: param_val.first.parameterValue.name == value_name1)

    if parameter2 != ANY_PARAM:
        filter_list.append(lambda param_val: param_val.second.parameter == parameter2)

    if value_name2 != ANY_NAME:
        filter_list.append(lambda param_val: param_val.second.parameterValue.name == value_name2)

    range_ver1 = _create_version_range(
        value_min_version1,
        value_min_version1_inclusive,
        value_max_version1,
        value_max_version1_inclusive,
    )
    filter_list.append(lambda param_val: param_val.first.parameterValue.version in range_ver1)

    range_ver2 = _create_version_range(
        value_min_version2,
        value_min_version2_inclusive,
        value_max_version2,
        value_max_version2_inclusive,
    )
    filter_list.append(lambda param_val: param_val.second.parameterValue.version in range_ver2)

    def filter_func(param_value_pair: ParameterValuePair) -> bool:
        return_value = True

        for f in filter_list:
            return_value = return_value and f(param_value_pair)

        return not return_value

    len_before = len(parameter_value_pairs)
    bi_filter(parameter_value_pairs, removed_parameter_value_pairs, filter_func)

    if symmetric:
        remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter2,
            value_name2,
            value_min_version2,
            value_min_version2_inclusive,
            value_max_version2,
            value_max_version2_inclusive,
            parameter1,
            value_name1,
            value_min_version1,
            value_min_version1_inclusive,
            value_max_version1,
            value_max_version1_inclusive,
            symmetric=False,
        )

    return len_before != len(parameter_value_pairs)


# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments
@typechecked
def remove_parameter_value_pairs(  # pylint: disable=too-many-arguments
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    parameter1: Parameter = ANY_PARAM,
    value_name1: ValueName = ANY_NAME,
    value_version1: Union[int, float, str] = ANY_VERSION,
    parameter2: Parameter = ANY_PARAM,
    value_name2: ValueName = ANY_NAME,
    value_version2: Union[int, float, str] = ANY_VERSION,
    symmetric: bool = True,
) -> bool:
    """Removes a parameter-value-pair from a list based on the specified search criteria. A
    parameter-value pair must match all specified search criteria to be removed if none of the
    criteria is `ANY_*`. If a criterion is `ANY_*`, it is ignored and it is always a match.

    Use `remove_parameter_value_pairs()` if you want to remove a single version from the first
    and/or second parameter value. If you want to remove a range of versions, use
    `remove_parameter_value_pairs_ranges()`.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): list where parameter-value-pairs will be
            removed
        removed_parameter_value_pairs (List[ParameterValuePair]): list where removed
            parameter-value-pairs will be stored
        parameter1 (Parameter, optional): Name of the first parameter. Defaults to ANY_PARAM.
        value_name1 (ValueName, optional): Name of the first value-name. Defaults to ANY_NAME.
        value_version1 (Union[int, float, str], optional): Name of the first value-version. Needs to
            be parsable to `Packaging.version.Version`. Defaults to ANY_VERSION.
        parameter2 (Parameter, optional): Name of the second parameter. Defaults to ANY_PARAM.
        value_name2 (ValueName, optional): Name of the second value-name. Defaults to ANY_NAME.
        value_version2 (Union[int, float, str], optional): Name of the second value-name. Needs to
            be parsable to `Packaging.version.Version`. Defaults to ANY_VERSION.
        symmetric (bool, optional): If symmetric is true, it does not matter whether a group of
            parameters, value-name and value-version was found in the first or second
            parameter-value. If false, it is taken into account whether the search criterion was
            found in the first or second parameter value. Defaults to True.

    Returns:
        bool: Return True, if parameter-value-pair was removed.
    """
    for v in (value_version1, value_version2):
        if v != ANY_VERSION:
            packaging.version.Version(str(v))

    return remove_parameter_value_pairs_ranges(
        parameter_value_pairs=parameter_value_pairs,
        removed_parameter_value_pairs=removed_parameter_value_pairs,
        parameter1=parameter1,
        value_name1=value_name1,
        value_min_version1=value_version1,
        value_min_version1_inclusive=True,
        value_max_version1=value_version1,
        value_max_version1_inclusive=True,
        parameter2=parameter2,
        value_name2=value_name2,
        value_min_version2=value_version2,
        value_min_version2_inclusive=True,
        value_max_version2=value_version2,
        value_max_version2_inclusive=True,
        symmetric=symmetric,
    )
