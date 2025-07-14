"""Filter rules to remove combinations which has to do with HIP"""

from typing import List, Dict, Callable
from packaging.specifiers import SpecifierSet
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValueSingle, ParameterValuePair
from bashi.utils import remove_parameter_value_pairs, remove_parameter_value_pairs_ranges
from bashi.versions import UBUNTU_HIP_VERSION_RANGE


def _remove_unsupported_compiler_for_hip_backend(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where the hip backend is enabled and the compiler is not hipcc.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_name in COMPILERS:
        if compiler_name != HIPCC:
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                remove_parameter_value_pairs(
                    parameter_value_pairs,
                    removed_parameter_value_pairs,
                    parameter1=compiler_type,
                    value_name1=compiler_name,
                    value_version1=ANY_VERSION,
                    parameter2=ALPAKA_ACC_GPU_HIP_ENABLE,
                    value_name2=ALPAKA_ACC_GPU_HIP_ENABLE,
                    value_version2=ON,
                )


def _remove_disabled_hip_backend_for_hipcc(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where the hipcc is the compiler and the hip backend is disabled.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=HIPCC,
            value_version1=ANY_VERSION,
            parameter2=ALPAKA_ACC_GPU_HIP_ENABLE,
            value_name2=ALPAKA_ACC_GPU_HIP_ENABLE,
            value_version2=OFF,
        )


def _remove_enabled_sycl_backend_for_hipcc(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where the hipcc is the compiler and the sycl backend is enabled.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=HIPCC,
            value_version1=ANY_VERSION,
            parameter2=ALPAKA_ACC_SYCL_ENABLE,
            value_name2=ALPAKA_ACC_SYCL_ENABLE,
            value_version2=ON,
        )


def _remove_enabled_cuda_backend_for_hipcc(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where the hipcc is the compiler and the cuda backend is enabled.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=HIPCC,
            parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_name2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_min_version2=OFF,
            value_min_version2_inclusive=False,
        )


def _remove_unsupported_hipcc_ubuntu_combinations(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs where HIPCC does not support a specific Ubuntu version

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
        removed_parameter_value_pairs (List[ParameterValuePair): list with removed
            parameter-value-pairs
    """
    ub_hip_ranges: Dict[ValueVersion, SpecifierSet] = {}
    for ub_hip in UBUNTU_HIP_VERSION_RANGE:
        ub_hip_ranges[ub_hip.ubuntu] = ub_hip.hip_range

    tmp_parameter_value_pairs: List[ParameterValuePair] = []

    def is_remove(param_val1: ParameterValueSingle, param_val2: ParameterValueSingle) -> bool:
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            if (
                param_val1.parameter == UBUNTU
                and param_val2.parameter == compiler_type
                and param_val2.parameterValue.name == HIPCC
            ):
                if param_val1.parameterValue.version not in ub_hip_ranges:
                    return True

                if (
                    param_val2.parameterValue.version
                    not in ub_hip_ranges[param_val1.parameterValue.version]
                ):
                    return True

        return False

    for param_val in parameter_value_pairs:
        if is_remove(param_val.first, param_val.second) or is_remove(
            param_val.second, param_val.first
        ):
            removed_parameter_value_pairs.append(param_val)
        else:
            tmp_parameter_value_pairs.append(param_val)

    parameter_value_pairs[:] = tmp_parameter_value_pairs


def _remove_unsupported_hip_backend_ubuntu_combinations(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    runtime_infos: Dict[str, Callable[..., bool]],
):
    """Remove all pairs Ubuntu HIP backend pairs, which are not support for the given runtime info.
    Depending on the given HIPCC versions in the input parameter-value-matrix it is possible that
    there is no HIP version which can be installed a specific Ubuntu version.

    For example HIP 6.0 until 6.2 can be installed on Ubuntu 22.04. If no HIPCC 6.0 - 6.2 is in the
    input matrix, there is no possibility to install HIP on Ubuntu 22.04

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
        removed_parameter_value_pairs (List[ParameterValuePair): list with removed
            parameter-value-pairs
        runtime_infos: Dict[str, Callable[..., bool]]: Dict of functors which checks if the given
        parameter-values (combinations) are valid. This filter uses the functor
            RT_AVAILABLE_HIP_SDK_UBUNTU_VER.
    """
    if RT_AVAILABLE_HIP_SDK_UBUNTU_VER not in runtime_infos:
        return

    tmp_parameter_value_pairs: List[ParameterValuePair] = []

    def is_remove(param_val1: ParameterValueSingle, param_val2: ParameterValueSingle) -> bool:
        if (
            param_val1.parameter == ALPAKA_ACC_GPU_HIP_ENABLE
            and param_val1.parameterValue.version == ON_VER
            and param_val2.parameter == UBUNTU
            and not runtime_infos[RT_AVAILABLE_HIP_SDK_UBUNTU_VER](
                param_val2.parameterValue.version
            )
        ):
            return True

        return False

    for param_val in parameter_value_pairs:
        if is_remove(param_val.first, param_val.second) or is_remove(
            param_val.second, param_val.first
        ):
            removed_parameter_value_pairs.append(param_val)
        else:
            tmp_parameter_value_pairs.append(param_val)

    parameter_value_pairs[:] = tmp_parameter_value_pairs
