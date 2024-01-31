"""Different helper functions for bashi"""

from typing import Dict, List, IO
from collections import OrderedDict
import dataclasses
import sys
from typeguard import typechecked
from bashi.types import (
    Parameter,
    ParameterValue,
    ParameterValueTuple,
    ParameterValuePair,
    ParameterValueMatrix,
    CombinationList,
    FilterFunction,
)


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
def get_expected_parameter_value_pairs(
    parameter_matrix: ParameterValueMatrix,
) -> List[ParameterValuePair]:
    """Takes parameter-value-matrix an creates a list of all expected parameter-values-pairs.
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
            param_val_pair: ParameterValuePair = OrderedDict()
            param_val_pair[v1_parameter] = ParameterValue(v1_name, v1_version)
            param_val_pair[v2_parameter] = ParameterValue(v2_name, v2_version)
            expected_pairs.append(param_val_pair)


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
        param1, param_val1 = list(ex_param_val_pair.items())[0]
        param2, param_val2 = list(ex_param_val_pair.items())[1]
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
