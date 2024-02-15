"""Functions to generate the combination-list"""

from typing import Dict, List
from collections import OrderedDict
import copy
import packaging.version as pkv

from covertable import make  # type: ignore

from bashi.types import (
    Parameter,
    ParameterValue,
    ParameterValueMatrix,
    FilterFunction,
    Combination,
    CombinationList,
)
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_chain import get_default_filter_chain


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
    # use local version to do not modify parameter_value_matrix
    local_param_val_mat = copy.deepcopy(parameter_value_matrix)

    filter_chain = get_default_filter_chain(custom_filter)

    def host_compiler_filter(param_val: ParameterValue) -> bool:
        # Rule: n1
        # remove nvcc as host compiler
        if param_val.name == NVCC:
            return False
        # Rule: v5
        # remove clang-cuda older than 14
        if param_val.name == CLANG_CUDA and param_val.version < pkv.parse("14"):
            return False

        return True

    def device_compiler_filter(param_val: ParameterValue) -> bool:
        # Rule: v5
        # remove clang-cuda older than 14
        if param_val.name == CLANG_CUDA and param_val.version < pkv.parse("14"):
            return False

        return True

    pre_filters = {HOST_COMPILER: host_compiler_filter, DEVICE_COMPILER: device_compiler_filter}

    # some filter rules requires that specific parameter-values are already removed from the
    # parameter-value-matrix
    # otherwise the covertable library throws an error
    for param, filter_func in pre_filters.items():
        if param in local_param_val_mat:
            local_param_val_mat[param] = list(filter(filter_func, local_param_val_mat[param]))

    comb_list: CombinationList = []

    all_pairs: List[Dict[Parameter, ParameterValue]] = make(
        factors=local_param_val_mat,
        length=2,
        pre_filter=filter_chain,
    )  # type: ignore

    # convert List[Dict[Parameter, ParameterValue]] to CombinationList
    for all_pair in all_pairs:
        tmp_comb: Combination = OrderedDict({})
        # covertable does not keep the ordering of the parameters
        # therefore we sort it
        for param in local_param_val_mat.keys():
            tmp_comb[param] = all_pair[param]
        comb_list.append(tmp_comb)

    return comb_list
