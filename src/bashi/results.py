"""Create list of expected parameter-value-pairs respecting bashi filter rules"""

from typing import List, Tuple, Dict, Callable
from typeguard import typechecked
from bashi.types import ParameterValuePair, ParameterValueMatrix
from bashi.utils import (
    get_expected_parameter_value_pairs,
    remove_parameter_value_pairs,
    remove_parameter_value_pairs_ranges,
    bi_filter,
)
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import COMPILERS
from bashi.result_modules.cuda_support import remove_cuda_specific_parameter_value_pairs
from bashi.result_modules.hip_support import remove_hip_specific_parameter_value_pairs
from bashi.result_modules.cxx_compiler_support import remove_cxx_specific_parameter_value_pairs


@typechecked
def get_expected_bashi_parameter_value_pairs(
    parameter_matrix: ParameterValueMatrix,
    runtime_infos: Dict[str, Callable[..., bool]],
) -> Tuple[List[ParameterValuePair], List[ParameterValuePair]]:
    """Takes parameter-value-matrix and creates a list of all expected parameter-values-pairs
    allowed by the bashi library. First it generates a complete list of parameter-value-pairs and
    then it removes all pairs that are not allowed by filter rules.

    Args:
        parameter_matrix (ParameterValueMatrix): matrix of parameter values

    Returns:
        List[ParameterValuePair]: list of all parameter-value-pairs supported by bashi
    """
    param_val_pair_list = get_expected_parameter_value_pairs(parameter_matrix)
    removed_param_val_pair_list: List[ParameterValuePair] = []

    _remove_different_compiler_names(param_val_pair_list, removed_param_val_pair_list)
    _remove_different_compiler_versions(param_val_pair_list, removed_param_val_pair_list)
    _remove_enabled_hip_and_sycl_backend_at_same_time(
        param_val_pair_list, removed_param_val_pair_list
    )
    _remove_enabled_cuda_backend_for_enabled_hip_backend(
        param_val_pair_list, removed_param_val_pair_list
    )
    _remove_unsupported_compiler_for_sycl_backend(param_val_pair_list, removed_param_val_pair_list)
    _remove_disabled_sycl_backend_for_icpx(param_val_pair_list, removed_param_val_pair_list)
    _remove_enabled_hip_backend_for_icpx(param_val_pair_list, removed_param_val_pair_list)
    _remove_enabled_cuda_backend_for_icpx(param_val_pair_list, removed_param_val_pair_list)
    _remove_enabled_cuda_backend_for_enabled_sycl_backend(
        param_val_pair_list, removed_param_val_pair_list
    )
    _remove_unsupported_gcc_versions_for_ubuntu2004(
        param_val_pair_list, removed_param_val_pair_list
    )

    remove_cxx_specific_parameter_value_pairs(param_val_pair_list, removed_param_val_pair_list)
    remove_cuda_specific_parameter_value_pairs(
        param_val_pair_list, removed_param_val_pair_list, runtime_infos
    )
    remove_hip_specific_parameter_value_pairs(
        param_val_pair_list, removed_param_val_pair_list, runtime_infos
    )
    return (param_val_pair_list, removed_param_val_pair_list)


def _remove_different_compiler_names(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all combinations, where host and device compiler name are different except the device
    compiler name is nvcc.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    # remove all combinations, where host and device compiler name are different except the device
    # compiler name is nvcc
    for host_compiler_name in set(COMPILERS) - set([NVCC]):
        for device_compiler_name in set(COMPILERS) - set([NVCC]):
            if host_compiler_name != device_compiler_name:
                remove_parameter_value_pairs_ranges(
                    parameter_value_pairs,
                    removed_parameter_value_pairs,
                    parameter1=HOST_COMPILER,
                    value_name1=host_compiler_name,
                    parameter2=DEVICE_COMPILER,
                    value_name2=device_compiler_name,
                )


def _remove_different_compiler_versions(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all combinations, where host and device compiler name are equal and versions are
    different except the compiler name is nvcc.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """

    def filter_function(param_val_pair: ParameterValuePair) -> bool:
        param_names = (HOST_COMPILER, DEVICE_COMPILER)
        compiler_names = set(COMPILERS) - set([NVCC])

        if (
            param_val_pair.first.parameter in param_names
            and param_val_pair.second.parameter in param_names
            and param_val_pair.first.parameterValue.name
            == param_val_pair.second.parameterValue.name
            and param_val_pair.first.parameterValue.name in compiler_names
            and param_val_pair.first.parameterValue.version
            != param_val_pair.second.parameterValue.version
        ):
            return False

        return True

    bi_filter(parameter_value_pairs, removed_parameter_value_pairs, filter_function)


def _remove_enabled_hip_and_sycl_backend_at_same_time(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where the HIP and the sycl backend are enabled at the same time.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    remove_parameter_value_pairs(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=ALPAKA_ACC_GPU_HIP_ENABLE,
        value_name1=ALPAKA_ACC_GPU_HIP_ENABLE,
        value_version1=ON,
        parameter2=ALPAKA_ACC_SYCL_ENABLE,
        value_name2=ALPAKA_ACC_SYCL_ENABLE,
        value_version2=ON,
    )


def _remove_enabled_cuda_backend_for_enabled_hip_backend(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where the HIP and the CUDA backend is enabled at the same time.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    remove_parameter_value_pairs_ranges(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=ALPAKA_ACC_GPU_HIP_ENABLE,
        value_name1=ALPAKA_ACC_GPU_HIP_ENABLE,
        value_min_version1=OFF,
        value_min_version1_inclusive=False,
        parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_name2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_min_version2=OFF,
        value_min_version2_inclusive=False,
    )


def _remove_unsupported_compiler_for_sycl_backend(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where the compiler does not support the SYCL backend.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_name in COMPILERS:
        if compiler_name != ICPX:
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                remove_parameter_value_pairs(
                    parameter_value_pairs,
                    removed_parameter_value_pairs,
                    parameter1=compiler_type,
                    value_name1=compiler_name,
                    value_version1=ANY_VERSION,
                    parameter2=ALPAKA_ACC_SYCL_ENABLE,
                    value_name2=ALPAKA_ACC_SYCL_ENABLE,
                    value_version2=ON,
                )


def _remove_disabled_sycl_backend_for_icpx(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where the ICPX is the compiler and the SYCL backend is disabled.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=ICPX,
            value_version1=ANY_VERSION,
            parameter2=ALPAKA_ACC_SYCL_ENABLE,
            value_name2=ALPAKA_ACC_SYCL_ENABLE,
            value_version2=OFF,
        )


def _remove_enabled_hip_backend_for_icpx(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where ICPX is the compiler and the HIP backend is enabled.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=ICPX,
            value_version1=ANY_VERSION,
            parameter2=ALPAKA_ACC_GPU_HIP_ENABLE,
            value_name2=ALPAKA_ACC_GPU_HIP_ENABLE,
            value_version2=ON,
        )


def _remove_enabled_cuda_backend_for_icpx(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where ICPX is the compiler and the CUDA backend is enabled.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=ICPX,
            parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_name2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_min_version2=OFF,
            value_min_version2_inclusive=False,
        )


def _remove_enabled_cuda_backend_for_enabled_sycl_backend(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where the SYCL and the CUDA backend is enabled at the same time.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    remove_parameter_value_pairs_ranges(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=ALPAKA_ACC_SYCL_ENABLE,
        value_name1=ALPAKA_ACC_SYCL_ENABLE,
        value_min_version1=ON,
        value_max_version1=ON,
        parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_name2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_min_version2=OFF,
        value_min_version2_inclusive=False,
    )


def _remove_unsupported_gcc_versions_for_ubuntu2004(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove pairs where GCC version 6 and older is used with Ubuntu 20.04 or newer.

    Args:

    parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=GCC,
            value_max_version1=6,
            parameter2=UBUNTU,
            value_name2=UBUNTU,
            value_min_version2="20.04",
        )
