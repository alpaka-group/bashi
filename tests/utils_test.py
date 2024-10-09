"""Different helper functions for bashi tests."""

import unittest
from typing import List, Union, Tuple, Callable, TypeAlias
from collections import OrderedDict
import packaging.version as pkv
from typeguard import typechecked
from bashi.types import (
    Parameter,
    ParameterValue,
    ParameterValuePair,
    ValueName,
)
from bashi.utils import create_parameter_value_pair
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


def parse_param_val(param_val: Tuple[ValueName, Union[str, int, float]]) -> ParameterValue:
    """Parse a single tuple to a parameter-values.

    Args:
        param_val (Tuple[ValueName, Union[str, int, float]]): Tuple to parse

    Returns:
        ParameterValue: parsed ParameterValue
    """
    val_name, val_version = param_val
    return ParameterValue(val_name, pkv.parse(str(val_version)))


def parse_param_vals(
    param_vals: List[Tuple[ValueName, Union[str, int, float]]]
) -> List[ParameterValue]:
    """Parse a list of tuples to a list of parameter-values.

    Args:
        param_vals (List[Tuple[ValueName, Union[str, int, float]]]): List to parse

    Returns:
        List[ParameterValue]: List of parameter-values
    """
    parsed_list: List[ParameterValue] = []

    for param_val in param_vals:
        parsed_list.append(parse_param_val(param_val))

    return parsed_list


def parse_expected_val_pairs(
    input_list: List[OrderedDict[Parameter, Tuple[ValueName, Union[str, int, float]]]]
) -> List[ParameterValuePair]:
    """Parse list of expected parameter-values to the correct type.

    Note: please use parse_expected_val_pairs2, which has a better interface

    Args:
        input_list (List[OrderedDict[Parameter, Tuple[ValueName, Union[str, int, float]]]]):
            List to parse

    Returns:
        List[ParameterValuePair]: Parsed parameter-type list
    """
    expected_val_pairs: List[ParameterValuePair] = []

    for param_val_pair in input_list:
        if len(param_val_pair) != 2:
            raise RuntimeError(f"{param_val_pair}\ninput_list needs to have two entries")

        it = iter(param_val_pair.items())
        param1, param_val1 = next(it)
        val_name1, val_version1 = param_val1
        param2, param_val2 = next(it)
        val_name2, val_version2 = param_val2

        expected_val_pairs.append(
            create_parameter_value_pair(
                param1,
                val_name1,
                val_version1,
                param2,
                val_name2,
                val_version2,
            )
        )

    return expected_val_pairs


RegularParsableParameterValue: TypeAlias = Tuple[str, str, Union[str, int, float]]
CompilerParsableParameterValue: TypeAlias = Tuple[str, str, Union[str, int, float]]
DefaultParsableParameterValue: TypeAlias = Tuple[str, Union[str, int, float]]
ParsableParameterValue: TypeAlias = Union[
    DefaultParsableParameterValue, CompilerParsableParameterValue
]


def parse_expected_val_pairs2(
    input_list: List[Tuple[ParsableParameterValue, ParsableParameterValue]]
) -> List[ParameterValuePair]:
    """Parse list of expected parameter-values to the correct type.

    Args:
        input_list (List[Tuple[ParsableParameterValue, ParsableParameterValue]]):
            e.g.:
            parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 12), (UBUNTU, 22.04)),
                ((HOST_COMPILER, CLANG_CUDA, 14), (CMAKE, "3.19")),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, "10.1")),
                ((UBUNTU, "22.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, "10.1")),
            ])

    Returns:
        List[ParameterValuePair]: Parsed parameter-type list
    """
    expected_val_pairs: List[ParameterValuePair] = []
    for pair_number, input_pair in enumerate(input_list):
        if not isinstance(input_pair, tuple):
            raise TypeError(f"{input_pair}\ninput_list[{pair_number}] is not a tuple")
        if len(input_pair) != 2:
            raise ValueError(
                f"{input_pair}\ninput_list[{pair_number}] needs to be a tuple with two entries."
            )

        regular_entry_pair: List[RegularParsableParameterValue] = []
        for entry_number, input_entry in enumerate(input_pair):
            if input_entry[0] not in PARAMETERS:
                raise ValueError(
                    f"input_list[{pair_number}][{entry_number}][0] {input_entry}\n"
                    "Parameter is unknown.\n"
                )

            if input_entry[0] in (HOST_COMPILER, DEVICE_COMPILER):
                if len(input_entry) != 3:
                    raise ValueError(
                        f"input_list[{pair_number}][{entry_number}] {input_entry}\n"
                        "First value is HOST_COMPILER or DEVICE_COMPILER.\n"
                        "Therefore the tuple needs to contain three entries:"
                        "\n(<HOST_COMPILER|DEVICE_COMPILER>, <value-name>, <value-version>)"
                    )
                if input_entry[1] not in COMPILERS:
                    raise ValueError(
                        f"input_list[{pair_number}][{entry_number}][1] {input_entry}\n"
                        "Compiler is unknown.\n"
                    )
                regular_entry_pair.append(input_entry)
            else:
                if len(input_entry) != 2:
                    raise ValueError(
                        f"input_list[{pair_number}][{entry_number}] {input_entry}\n"
                        "The tuple needs to contain two entries:"
                        "\n(<parameter>, <value-version>)"
                    )
                regular_entry_pair.append((input_entry[0], input_entry[0], input_entry[1]))

        expected_val_pairs.append(
            create_parameter_value_pair(
                regular_entry_pair[0][0],
                regular_entry_pair[0][1],
                regular_entry_pair[0][2],
                regular_entry_pair[1][0],
                regular_entry_pair[1][1],
                regular_entry_pair[1][2],
            )
        )

    return expected_val_pairs


def create_diff_parameter_value_pairs(
    given_result: List[ParameterValuePair], expected_result: List[ParameterValuePair]
) -> str:
    """Returns a string for a readable output, if two lists of parameter-value-pairs are different.

    Args:
        given_result (List[ParameterValuePair]): Results from the test
        expected_result (List[ParameterValuePair]): Expected results

    Returns:
        str: Output string
    """
    output = f"\ngiven ({len(given_result)} elements):\n"
    for g_result in sorted(given_result):
        output += (
            f"  {g_result.first.parameter}="
            f"{g_result.first.parameterValue.name} {g_result.first.parameterValue.version} + "
            f"{g_result.second.parameter}="
            f"{g_result.second.parameterValue.name} {g_result.second.parameterValue.version}\n"
        )

    output += f"expected ({len(expected_result)} elements):\n"

    for e_result in sorted(expected_result):
        output += (
            f"  {e_result.first.parameter}="
            f"{e_result.first.parameterValue.name} {e_result.first.parameterValue.version} + "
            f"{e_result.second.parameter}="
            f"{e_result.second.parameterValue.name} {e_result.second.parameterValue.version}\n"
        )

    return output


@typechecked
def default_remove_test(
    function: Callable[[List[ParameterValuePair], List[ParameterValuePair]], None],
    test_parameter_value_pairs: List[ParameterValuePair],
    expected_results: List[ParameterValuePair],
    test_self: unittest.TestCase,
):
    """Test template for sub-functions of the get_expected_bashi_parameter_value_pairs() function.
    Takes a function, an input parameter-value-pair list and a list of expected
    parameter-value-pairs and compares the result of the function with the expected result. Also
    checks the unexpected result.

    Args:
        function (Callable[[List[ParameterValuePair], List[ParameterValuePair]], None]): Function to
            test
        test_parameter_value_pairs (List[ParameterValuePair]): Parameter-value-pairs list to filter
        expected_results (List[ParameterValuePair]): Expected result list after the function was
            applied on the input list
        test_self (unittest.TestCase): The function needs to be called in an unittest function. To
            allow to use the features of the unittest module, the caller function needs to pass the
            self parameter.
    """
    expected_results.sort()
    unexpected_results: List[ParameterValuePair] = sorted(
        list(set(test_parameter_value_pairs) - set(expected_results))
    )

    unexpected_test_param_value_pairs: List[ParameterValuePair] = []
    function(test_parameter_value_pairs, unexpected_test_param_value_pairs)

    test_parameter_value_pairs.sort()
    unexpected_test_param_value_pairs.sort()

    test_self.assertEqual(
        test_parameter_value_pairs,
        expected_results,
        create_diff_parameter_value_pairs(test_parameter_value_pairs, expected_results),
    )
    test_self.assertEqual(
        unexpected_test_param_value_pairs,
        unexpected_results,
        create_diff_parameter_value_pairs(unexpected_test_param_value_pairs, unexpected_results),
    )
