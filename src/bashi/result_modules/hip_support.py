"""Filter rules to remove combinations which has to do with HIP"""

from typing import List, Dict, Callable
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValuePair
from bashi.utils import remove_parameter_value_pairs, remove_parameter_value_pairs_ranges
from bashi.versions import UBUNTU_HIP_VERSION_RANGE
from bashi.result_modules.sdk_helper import (
    remove_unsupported_sdk_ubuntu_combinations,
    remove_runtime_not_available_ubuntu_versions,
)


def remove_hip_specific_parameter_value_pairs(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    runtime_infos: Dict[str, Callable[..., bool]],
):
    """Apply several filter functions to remove invalid HIP related parameter-value-pairs.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
        removed_parameter_value_pairs (List[ParameterValuePair): list with removed
            parameter-value-pairs
        runtime_infos: Dict[str, Callable[..., bool]]: Dict of functors which checks if the given
        parameter-values (combinations) are valid.
    """
    _remove_unsupported_compiler_for_hip_backend(
        parameter_value_pairs, removed_parameter_value_pairs
    )
    _remove_disabled_hip_backend_for_hipcc(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_enabled_sycl_backend_for_hipcc(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_enabled_cuda_backend_for_hipcc(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_unsupported_hipcc_ubuntu_combinations(
        parameter_value_pairs, removed_parameter_value_pairs
    )
    _remove_runtime_unsupported_hip_backend_ubuntu_combinations(
        parameter_value_pairs, removed_parameter_value_pairs, runtime_infos
    )


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
    for sycl_backend in ONE_API_BACKENDS:
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            remove_parameter_value_pairs(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                parameter1=compiler_type,
                value_name1=HIPCC,
                value_version1=ANY_VERSION,
                parameter2=sycl_backend,
                value_name2=sycl_backend,
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
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_unsupported_sdk_ubuntu_combinations(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            compiler_type,
            HIPCC,
            UBUNTU_HIP_VERSION_RANGE,
        )


def _remove_runtime_unsupported_hip_backend_ubuntu_combinations(
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

    remove_runtime_not_available_ubuntu_versions(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        runtime_infos[RT_AVAILABLE_HIP_SDK_UBUNTU_VER],
        ALPAKA_ACC_GPU_HIP_ENABLE,
    )
