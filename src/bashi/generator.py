"""Functions to generate the combination-list"""

from typing import Dict, List, Callable
from collections import OrderedDict

from covertable import make  # type: ignore

from bashi.types import (
    Parameter,
    ParameterValue,
    ParameterValueMatrix,
    Combination,
    CombinationList,
)
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter import FilterBase
from bashi.filter_chain import get_default_filter_chain, FilterChain


def generate_combination_list(
    parameter_value_matrix: ParameterValueMatrix,
    custom_filter: FilterBase = FilterBase(),
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
    runtime_infos: Dict[str, Callable[..., bool]] = {}

    filter_chain: FilterChain = get_default_filter_chain(
        runtime_infos=runtime_infos, custom_filter=custom_filter
    )

    comb_list: CombinationList = []

    all_pairs: List[Dict[Parameter, ParameterValue]] = make(
        factors=parameter_value_matrix,
        length=2,
        pre_filter=filter_chain,
    )  # type: ignore

    # convert List[Dict[Parameter, ParameterValue]] to CombinationList
    for all_pair in all_pairs:
        tmp_comb: Combination = OrderedDict({})
        # covertable does not keep the ordering of the parameters
        # therefore we sort it
        for param in parameter_value_matrix.keys():
            tmp_comb[param] = all_pair[param]
        comb_list.append(tmp_comb)

    return comb_list
