"""Functions to generate the combination-list"""

from typing import Dict, List, Callable, cast
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
from bashi.runtime_info import get_sdk_supporting_ubuntus
from bashi.versions import UBUNTU_HIP_VERSION_RANGE, UBUNTU_CUDA_VERSION_RANGE


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
        ubuntus: List[ValueVersion] = [
            param_val.version for param_val in parameter_value_matrix[UBUNTU]
        ]
        if len(ubuntus) > 0:
            for sdk_name, version_range, rt_func_name in [
                (
                    HIPCC,
                    UBUNTU_HIP_VERSION_RANGE,
                    RT_AVAILABLE_HIP_SDK_UBUNTU_VER,
                ),
                (
                    NVCC,
                    UBUNTU_CUDA_VERSION_RANGE,
                    RT_AVAILABLE_CUDA_SDK_UBUNTU_VER,
                ),
            ]:
                sdks: List[ValueVersion] = []
                for param_val in parameter_value_matrix[DEVICE_COMPILER]:
                    if param_val.name == sdk_name:
                        sdks.append(param_val.version)

                if len(sdks) > 0 and len(version_range) > 0:
                    runtime_infos[rt_func_name] = get_sdk_supporting_ubuntus(
                        ubuntus=ubuntus,
                        sdk_versions=sdks,
                        ubuntu_sdk_version_range=version_range,
                    )

    return runtime_infos


def generate_combination_list(
    parameter_value_matrix: ParameterValueMatrix,
    runtime_infos: Dict[str, Callable[..., bool]],
    custom_filter: FilterBase = FilterBase(),
    debug_print: FilterDebugMode = FilterDebugMode.OFF,
) -> CombinationList:
    """Generate combination-list from the parameter-value-matrix. The combination list contains
    all valid parameter-value-pairs at least one time.

    Args:
        parameter_value_matrix (ParameterValueMatrix): Input matrix with parameter and
            parameter-values.
        custom_filter (FilterFunction, optional): Custom filter function to extend bashi
            filters. Defaults is lambda _: True.
        debug_print (FilterDebugMode): Depending on the debug mode, print additional information
            for each row passing the filter function. Defaults to FilterDebugMode.OFF.
    Returns:
        CombinationList: combination-list
    """

    filter_chain: FilterChain = get_default_filter_chain(
        runtime_infos=runtime_infos, custom_filter=custom_filter
    )
    filter_chain.set_debug_print(debug_print)

    comb_list: CombinationList = []

    all_pairs = cast(
        List[Dict[Parameter, ParameterValue]],
        make(
            factors=parameter_value_matrix,
            length=2,
            pre_filter=filter_chain,
        ),
    )

    # convert List[Dict[Parameter, ParameterValue]] to CombinationList
    for all_pair in all_pairs:
        tmp_comb: Combination = OrderedDict({})
        # covertable does not keep the ordering of the parameters
        # therefore we sort it
        for param in parameter_value_matrix.keys():
            tmp_comb[param] = all_pair[param]
        comb_list.append(tmp_comb)

    return comb_list
