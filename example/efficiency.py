"""Calculates how many combinations are saved with pair-wise combination."""

from typing import List
import locale
from collections import OrderedDict
from dataclasses import dataclass
from bashi.types import Parameter, Combination, CombinationList, ParameterValueMatrix
from bashi.filter_chain import get_default_filter_chain
from bashi.generator import generate_combination_list, get_runtime_infos
from bashi.versions import get_parameter_value_matrix

# print numbers with dots or commas as thousand delimiter depending on the local settings
locale.setlocale(locale.LC_ALL, "")


@dataclass
class Integer:
    """Wrapper for python int, because it pass as value instead as reference."""

    value: int = 0

    def inc(self):
        """Increment value by 1."""
        self.value += 1


# I don't understand why None is better default value
def count_combination(  # pylint: disable=dangerous-default-value
    parameter_matrix: ParameterValueMatrix,
    counter: Integer,
    param_val_indices: List[int] = [],
):
    """Count number of valid combinations.

    Args:
        parameter_matrix (ParameterValueMatrix): Parameter-value-matrix which will be used to
            generate the combinations.
        counter (Integer): Stores the number of valid combinations.
        param_val_indices (List[int], optional): The algorithm is recursive. This list stores the
            index postion of each parameter-value in the order of the parameter. Defaults to [].
    """
    bashi_filter_chain = get_default_filter_chain()
    parameters: List[Parameter] = list(parameter_matrix.keys())
    comb: Combination = OrderedDict()
    for param_index, value_index in enumerate(param_val_indices):
        comb[parameters[param_index]] = parameter_matrix[parameters[param_index]][value_index]

    my_index = len(param_val_indices)
    for param_value_index, param_values in enumerate(parameter_matrix[parameters[my_index]]):
        comb[parameters[my_index]] = param_values
        if bashi_filter_chain(comb):
            if my_index + 1 < len(parameters):
                count_combination(
                    parameter_matrix, counter, param_val_indices + [param_value_index]
                )
            else:
                counter.inc()


if __name__ == "__main__":
    tmp = get_parameter_value_matrix()
    num_param = len(tmp.keys())

    param_matrix: ParameterValueMatrix = OrderedDict()
    for index, param in enumerate(tmp.keys()):
        if index >= num_param:
            break
        param_matrix[param] = tmp[param]

    parameter_list = list(param_matrix.keys())[:num_param]

    # looks like a pylint bug, because the value is not constant
    num_combinations = 1  # pylint: disable=invalid-name
    for param in parameter_list:
        num_combinations *= len(param_matrix[param])

    print(
        f"Cartesian product of all parameter-values with invalid combination:           "
        f"{num_combinations:n}"
    )

    num_combinations_dense_matrix = Integer(0)
    count_combination(param_matrix, num_combinations_dense_matrix)
    print(
        f"Cartesian product of all parameter-values with only valid combination:        "
        f"{num_combinations_dense_matrix.value:n}"
    )

    rt_info = get_runtime_infos(param_matrix)
    comb_list: CombinationList = generate_combination_list(
        parameter_value_matrix=param_matrix, runtime_infos=rt_info
    )

    print(
        f"pair wise combinations of all parameter-values with only valid combination:   "
        f"{len(comb_list):n}"
    )
    reduced_combinations_percent = (
        100 - (len(comb_list) / num_combinations_dense_matrix.value) * 100
    )
    print(f"reduced combinations: {reduced_combinations_percent:.2f}%")
