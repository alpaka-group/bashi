"""Different helper functions for bashi tests."""

from typing import List, Union, Tuple
from collections import OrderedDict
import packaging.version as pkv
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
            raise RuntimeError("input_list needs to have two entries")

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
