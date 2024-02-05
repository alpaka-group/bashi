"""Functions to generate the combination-list"""

from typing import Dict
from collections import OrderedDict

from allpairspy import AllPairs

from bashi.types import (
    Parameter,
    ParameterValueMatrix,
    FilterFunction,
    Combination,
    CombinationList,
)
from bashi.utils import get_default_filter_chain, FilterAdapter


def generate_combination_list(
    parameter_value_matrix: ParameterValueMatrix,
    custom_filter: FilterFunction = lambda _: True,
) -> CombinationList:
    """Generate combination-list from the parameter-value-matrix. The combination list contains
    all valid parameter-value-pairs at least one time.

    Args:
        parameter_value_matrix (ParameterValueMatrix): Input matrix with parameter and
        parameter-values.
        custom_filter (FilterFunction, optional): Custom filter function to extend bashi
        filters. Defaults is lambda _: True.
    Returns:
        CombinationList: combination-list
    """
    filter_chain = get_default_filter_chain(custom_filter)

    param_map: Dict[int, Parameter] = {}
    for index, key in enumerate(parameter_value_matrix.keys()):
        param_map[index] = key
    filter_adapter = FilterAdapter(param_map, filter_chain)

    comb_list: CombinationList = []

    # convert List[Pair] to CombinationList
    for all_pair in AllPairs(  # type: ignore
        parameters=parameter_value_matrix, n=2, filter_func=filter_adapter
    ):
        comb: Combination = OrderedDict()
        for index, param in enumerate(all_pair._fields):  # type: ignore
            comb[param] = all_pair[index]  # type: ignore
        comb_list.append(comb)

    return comb_list
