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
from bashi.runtime_info import get_hip_sdk_supporting_ubuntus
from bashi.versions import UBUNTU_HIP_VERSION_RANGE


def get_runtime_infos(
    parameter_value_matrix: ParameterValueMatrix,
) -> Dict[str, Callable[..., bool]]:
    """Get several runtime filter rules for the given input parameter-value-matrix

    Args:
        parameter_value_matrix (ParameterValueMatrix): parameter-value-matrix

    Returns:
        Dict[str, Callable[..., bool]]: Dict of filter functions
    """
    runtime_infos: Dict[str, Callable[..., bool]] = {}

    if UBUNTU in parameter_value_matrix and DEVICE_COMPILER in parameter_value_matrix:
        hipccs: List[ValueVersion] = []
        for param_val in parameter_value_matrix[DEVICE_COMPILER]:
            if param_val.name == HIPCC:
                hipccs.append(param_val.version)

        ubuntus: List[ValueVersion] = [
            param_val.version for param_val in parameter_value_matrix[UBUNTU]
        ]

        if len(hipccs) > 0 and len(ubuntus) > 0 and len(UBUNTU_HIP_VERSION_RANGE) > 0:
            runtime_infos[RT_AVAILABLE_HIP_SDK_UBUNTU_VER] = get_hip_sdk_supporting_ubuntus(
                ubuntus=ubuntus,
                hipccs=hipccs,
                ubuntu_hip_version_range=UBUNTU_HIP_VERSION_RANGE,
            )

    return runtime_infos


def generate_combination_list(
    parameter_value_matrix: ParameterValueMatrix,
    runtime_infos: Dict[str, Callable[..., bool]],
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
