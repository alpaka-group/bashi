"""Different helper functions for bashi tests."""

import unittest
from typing import List, Union, Tuple, Callable
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
