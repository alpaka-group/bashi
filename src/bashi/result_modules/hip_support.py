"""Filter rules to remove combinations which has to do with HIP"""

from typing import List
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValuePair
from bashi.utils import remove_parameter_value_pairs, remove_parameter_value_pairs_ranges


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


def _remove_all_rocm_images_older_than_ubuntu2004_based(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs where Ubuntu is older than 20.04 and the HIP backend is enabled or the host
    or device compiler is HIPCC.
    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    """
    remove_parameter_value_pairs_ranges(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=UBUNTU,
        value_name1=UBUNTU,
        value_max_version1="20.04",
        value_max_version1_inclusive=False,
        parameter2=ALPAKA_ACC_GPU_HIP_ENABLE,
        value_name2=ALPAKA_ACC_GPU_HIP_ENABLE,
        value_min_version2=OFF,
        value_min_version2_inclusive=False,
    )
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=UBUNTU,
            value_name1=UBUNTU,
            value_max_version1="20.04",
            value_max_version1_inclusive=False,
            parameter2=compiler_type,
            value_name2=HIPCC,
        )
