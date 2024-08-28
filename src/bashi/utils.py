"""Different helper functions for bashi"""

import dataclasses
import sys
from collections import OrderedDict
from typing import IO, Dict, List, Optional, Union, Callable

import packaging.version
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from typeguard import typechecked

from bashi.types import (
    CombinationList,
    FilterFunction,
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
    ALPAKA_ACC_SYCL_ENABLE: "bSYCL",
    CXX_STANDARD: "c++",
}


@dataclasses.dataclass
class FilterAdapter:
    """
    An adapter for the filter functions used by allpairspy to provide a better filter function
    interface.

    Independent of the type of `parameter` (in the bashi naming convention:
    parameter-value-matrix type) used as an argument of AllPairs.__init__(), allpairspy always
    passes the same row type to the filter function: List of parameter-values.
    Therefore, the parameter name is encoded in the position in the row list. This makes it
    much more difficult to write filter rules.

    The FilterAdapter transforms the list of parameter values into a parameter-value-tuple, which
    has the type OrderedDict[str, Tuple[str, Version]].

    This user writes a filter rule function with the expected line type
    OrderedDict[str, Tuple[str, Version]], creates a FunctionAdapter object with the functor as a
    parameter and passes the FunctionAdapter object to AllPairs.__init__().

    filter function example:

    def filter_function(row: OrderedDict[str, Tuple[str, Version]]):
        if (
            DEVICE_COMPILER in row
            and row[DEVICE_COMPILER][NAME] == NVCC
            and row[DEVICE_COMPILER][VERSION] < pkv.parse("12.0")
        ):
            return False
        return True

    Args:
            param_map (Dict[int, str]): The param_map maps the index position of a parameter to the
                parameter name. Assuming the parameter-value-matrix has the following keys:
                ["param1", "param2", "param3"], the param_map should look like this:
                {0: "param1", 1 : "param2", 2 : "param3"}.

            filter_func (Callable[[OrderedDict[str, Tuple[str, Version]]], bool]): The filter
                function used by allpairspy, see class doc string.
    """

    param_map: Dict[int, Parameter]
    filter_func: FilterFunction

    def __call__(self, row: List[ParameterValue]) -> bool:
        """The expected interface of allpairspy filter rule.
        Transform the type of row from List[Tuple[str, Version]] to
        [OrderedDict[str, Tuple[str, Version]]].

        Args:
            row (List[Tuple[str, Version]]): the parameter-value-tuple

        Returns:
            bool: Returns True, if the parameter-value-tuple is valid
        """
        ordered_row: ParameterValueTuple = OrderedDict()
        for index, param_name in enumerate(row):
            ordered_row[self.param_map[index]] = param_name
        return self.filter_func(ordered_row)


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


# pylint: disable=too-many-locals
@typechecked
def remove_parameter_value_pairs(  # pylint: disable=too-many-arguments
    parameter_value_pairs: List[ParameterValuePair],
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

    Args:
        parameter_value_pairs (List[ParameterValuePair]): list where parameter-value-pairs will be
            removed
        parameter1 (Parameter, optional): Name of the first parameter. Defaults to ANY_PARAM.
        value_name1 (ValueName, optional): Name of the first value-name. Defaults to ANY_NAME.
        value_version1 (Union[int, float, str], optional): Name of the first value-version. Either
            as a single version or as a version range that can be parsed into a
            `packaging.specifier.SpecifierSet`. If it is a version range, all versions that are not
            within this range are removed. Defaults to ANY_VERSION.
        parameter2 (Parameter, optional): Name of the second parameter. Defaults to ANY_PARAM.
        value_name2 (ValueName, optional): Name of the second value-name. Defaults to ANY_NAME.
        value_version2 (Union[int, float, str], optional): Name of the second value-name. Either as
            a single version or as a version range that can be parsed into a
            `packaging.specifier.SpecifierSet`. If it is a version range, all versions that are not
            within this range are removed. Defaults to ANY_VERSION.
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

    def is_specifier_set(version: Union[int, float, str]) -> bool:
        try:
            SpecifierSet(str(version))
            return True
        except InvalidSpecifier:
            return False

    if (
        value_version1 != ANY_VERSION
        and value_version2 != ANY_VERSION
        and is_specifier_set(value_version1)
        and is_specifier_set(value_version2)
    ):
        specifier_set_version1 = SpecifierSet(str(value_version1))
        specifier_set_version2 = SpecifierSet(str(value_version2))

        filter_list.append(
            lambda param_val: not (
                param_val.first.parameterValue.version in specifier_set_version1
                and param_val.second.parameterValue.version in specifier_set_version2
            )
        )

    else:
        if value_version1 != ANY_VERSION:
            try:
                specifier_set_version1 = SpecifierSet(str(value_version1))
                filter_list.append(
                    lambda param_val: not param_val.first.parameterValue.version
                    in specifier_set_version1
                )
            except InvalidSpecifier:
                parsed_value_version1 = packaging.version.parse(str(value_version1))
                filter_list.append(
                    lambda param_val: param_val.first.parameterValue.version
                    == parsed_value_version1
                )

        if value_version2 != ANY_VERSION:
            try:
                specifier_set_version2 = SpecifierSet(str(value_version2))
                filter_list.append(
                    lambda param_val: not param_val.second.parameterValue.version
                    in specifier_set_version2
                )
            except InvalidSpecifier:
                parsed_value_version2 = packaging.version.parse(str(value_version2))
                filter_list.append(
                    lambda param_val: param_val.second.parameterValue.version
                    == parsed_value_version2
                )

    def filter_func(param_value_pair: ParameterValuePair) -> bool:
        return_value = True

        for f in filter_list:
            return_value = return_value and f(param_value_pair)

        return not return_value

    len_before = len(parameter_value_pairs)
    parameter_value_pairs[:] = list(filter(filter_func, parameter_value_pairs))

    if symmetric:
        remove_parameter_value_pairs(
            parameter_value_pairs,
            parameter2,
            value_name2,
            value_version2,
            parameter1,
            value_name1,
            value_version1,
            symmetric=False,
        )

    return len_before != len(parameter_value_pairs)


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
            print(f"{ex_param_val_pair} is missing in combination list", file=output)
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
                    f"found unexpected parameter-value-pair {ex_param_val_pair} "
                    "in combination list",
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
    s = init

    nice_version: dict[packaging.version.Version, str] = {
        ON_VER: "ON",
        OFF_VER: "OFF",
    }

    for param, val in row.items():
        parameter_prefix = "" if not bashi_validate else "--"
        if param in [HOST_COMPILER, DEVICE_COMPILER]:
            s += (
                f"{parameter_prefix}{PARAMETER_SHORT_NAME.get(param, param)}="
                f"{PARAMETER_SHORT_NAME.get(val.name, val.name)}@"
                f"{nice_version.get(val.version, str(val.version))} "
            )
        else:
            s += (
                f"{parameter_prefix}{PARAMETER_SHORT_NAME.get(param, param)}="
                f"{nice_version.get(val.version, str(val.version))} "
            )
    print(s)
