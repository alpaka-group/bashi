"""Different helper functions for bashi"""

import dataclasses
import sys
from collections import OrderedDict
from typing import IO, Dict, List, Optional, Union

import packaging.version
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
)
from bashi.versions import COMPILERS, VERSIONS, NVCC_GCC_MAX_VERSION, NVCC_CLANG_MAX_VERSION
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


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
def remove_parameter_value_pair(
    to_remove: Union[ParameterValueSingle, ParameterValuePair],
    parameter_value_pairs: List[ParameterValuePair],
    all_versions: bool = False,
) -> bool:
    """Removes a parameter-value pair with one or two entries from the parameter-value-pair list. If
    the parameter-value-pair only has one parameter value, all parameter-value-pairs that contain
    the parameter value are removed.

    Args:
        to_remove (Union[ParameterValueSingle, ParameterValuePair]): Parameter-value-single or
            parameter-value-pair to remove
        param_val_pairs (List[ParameterValuePair]): List of parameter-value-pairs. Will be modified.
        all_versions (bool): If it is `True` and `to_remove` has type of `ParameterValuePair`,
            removes all parameter-value-pairs witch matches the value-names independent of the
            value-version. Defaults to False.
    Raises:
        RuntimeError: If `all_versions=True` and `to_remove` is a `ParameterValueSingle`

    Returns:
        bool: True if entry was removed from parameter_value_pairs
    """
    if isinstance(to_remove, ParameterValueSingle):
        if all_versions:
            raise RuntimeError("all_versions=True is not support for ParameterValueSingle")

        return _remove_single_entry_parameter_value_pair(to_remove, parameter_value_pairs)

    if all_versions:
        return _remove_parameter_value_pair_all_versions(to_remove, parameter_value_pairs)

    try:
        parameter_value_pairs.remove(to_remove)
        return True
    except ValueError:
        return False


@typechecked
def _remove_single_entry_parameter_value_pair(
    to_remove: ParameterValueSingle, param_val_pairs: List[ParameterValuePair]
) -> bool:
    len_before = len(param_val_pairs)

    def filter_function(param_val_pair: ParameterValuePair) -> bool:
        for param_val_entry in param_val_pair:
            if param_val_entry == to_remove:
                return False
        return True

    param_val_pairs[:] = list(filter(filter_function, param_val_pairs))

    return len_before != len(param_val_pairs)


@typechecked
def _remove_parameter_value_pair_all_versions(
    to_remove: ParameterValuePair, param_val_pairs: List[ParameterValuePair]
) -> bool:
    len_before = len(param_val_pairs)

    def filter_function(param_val_pair: ParameterValuePair) -> bool:
        if (
            param_val_pair.first.parameter == to_remove.first.parameter
            and param_val_pair.second.parameter == to_remove.second.parameter
            and param_val_pair.first.parameterValue.name == to_remove.first.parameterValue.name
            and param_val_pair.second.parameterValue.name == to_remove.second.parameterValue.name
        ):
            return False
        return True

    param_val_pairs[:] = list(filter(filter_function, param_val_pairs))

    return len_before != len(param_val_pairs)


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


# pylint: disable=too-many-branches
@typechecked
def get_expected_bashi_parameter_value_pairs(
    parameter_matrix: ParameterValueMatrix,
) -> List[ParameterValuePair]:
    """Takes parameter-value-matrix and creates a list of all expected parameter-values-pairs
    allowed by the bashi library. First it generates a complete list of parameter-value-pairs and
    then it removes all pairs that are not allowed by filter rules.

    Args:
        parameter_matrix (ParameterValueMatrix): matrix of parameter values

    Returns:
        List[ParameterValuePair]: list of all parameter-value-pairs supported by bashi
    """
    param_val_pair_list = get_expected_parameter_value_pairs(parameter_matrix)

    extend_versions = VERSIONS.copy()
    extend_versions[CLANG_CUDA] = extend_versions[CLANG]

    # remove all combinations where nvcc is device compiler and the host compiler is not gcc or
    # clang
    for compiler_name in set(COMPILERS) - set([GCC, CLANG, NVCC]):
        remove_parameter_value_pair(
            to_remove=create_parameter_value_pair(
                HOST_COMPILER, compiler_name, 0, DEVICE_COMPILER, NVCC, 0
            ),
            parameter_value_pairs=param_val_pair_list,
            all_versions=True,
        )

    # remove all combinations, where host and device compiler name are different except the device
    # compiler name is nvcc
    for host_compiler_name in set(COMPILERS) - set([NVCC]):
        for device_compiler_name in set(COMPILERS) - set([NVCC]):
            if host_compiler_name != device_compiler_name:
                remove_parameter_value_pair(
                    to_remove=create_parameter_value_pair(
                        HOST_COMPILER,
                        host_compiler_name,
                        0,
                        DEVICE_COMPILER,
                        device_compiler_name,
                        0,
                    ),
                    parameter_value_pairs=param_val_pair_list,
                    all_versions=True,
                )

    # remove all combinations, where host and device compiler version are different except the
    # compiler name is nvcc
    for compiler_name in set(COMPILERS) - set([NVCC]):
        for compiler_version1 in extend_versions[compiler_name]:
            for compiler_version2 in extend_versions[compiler_name]:
                if compiler_version1 != compiler_version2:
                    remove_parameter_value_pair(
                        to_remove=create_parameter_value_pair(
                            HOST_COMPILER,
                            compiler_name,
                            compiler_version1,
                            DEVICE_COMPILER,
                            compiler_name,
                            compiler_version2,
                        ),
                        parameter_value_pairs=param_val_pair_list,
                        all_versions=False,
                    )

    # remove all gcc version, which are to new for a specific nvcc version
    nvcc_versions = [packaging.version.parse(str(v)) for v in VERSIONS[NVCC]]
    nvcc_versions.sort()
    gcc_versions = [packaging.version.parse(str(v)) for v in VERSIONS[GCC]]
    gcc_versions.sort()
    for nvcc_version in nvcc_versions:
        for max_nvcc_clang_version in NVCC_GCC_MAX_VERSION:
            if nvcc_version >= max_nvcc_clang_version.nvcc:
                for clang_version in gcc_versions:
                    if clang_version > max_nvcc_clang_version.host:
                        remove_parameter_value_pair(
                            to_remove=create_parameter_value_pair(
                                HOST_COMPILER,
                                GCC,
                                clang_version,
                                DEVICE_COMPILER,
                                NVCC,
                                nvcc_version,
                            ),
                            parameter_value_pairs=param_val_pair_list,
                        )
                break

    clang_versions = [packaging.version.parse(str(v)) for v in VERSIONS[CLANG]]
    clang_versions.sort()

    # remove all clang version, which are to new for a specific nvcc version
    for nvcc_version in nvcc_versions:
        for max_nvcc_clang_version in NVCC_CLANG_MAX_VERSION:
            if nvcc_version >= max_nvcc_clang_version.nvcc:
                for clang_version in clang_versions:
                    if clang_version > max_nvcc_clang_version.host:
                        remove_parameter_value_pair(
                            to_remove=create_parameter_value_pair(
                                HOST_COMPILER,
                                CLANG,
                                clang_version,
                                DEVICE_COMPILER,
                                NVCC,
                                nvcc_version,
                            ),
                            parameter_value_pairs=param_val_pair_list,
                        )
                break

    # remove all pairs, where clang is host-compiler for nvcc 11.3, 11.4 and 11.5 as device compiler
    for nvcc_version in [packaging.version.parse(str(v)) for v in [11.3, 11.4, 11.5]]:
        for clang_version in clang_versions:
            remove_parameter_value_pair(
                to_remove=create_parameter_value_pair(
                    HOST_COMPILER,
                    CLANG,
                    clang_version,
                    DEVICE_COMPILER,
                    NVCC,
                    nvcc_version,
                ),
                parameter_value_pairs=param_val_pair_list,
            )

    return param_val_pair_list
