"""Create list of expected parameter-value-pairs respecting bashi filter rules"""

import copy
from typing import List, Optional
from typeguard import typechecked
from packaging.specifiers import SpecifierSet
from bashi.types import ParameterValue, ParameterValuePair, ParameterValueMatrix
from bashi.utils import (
    get_expected_parameter_value_pairs,
    remove_parameter_value_pairs,
)

from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import (
    COMPILERS,
    NVCC_GCC_MAX_VERSION,
    NVCC_CLANG_MAX_VERSION,
    CLANG_CUDA_MAX_CUDA_VERSION,
    NvccHostSupport,
    ClangCudaSDKSupport,
)


@typechecked
def get_expected_bashi_parameter_value_pairs(
    parameter_matrix: ParameterValueMatrix,
) -> List[ParameterValuePair]:
    """Takes parameter-value-matrix and creates a list of all expected parameter-values-pairs
    allowed by the bashi library. First it generates a complete list of parameter-value-pairs and
    then it removes all pairs that are not allowed by filter rules.

    Args:
        parameter_matrix (ParameterValueMatrix): matrix of parameter values

    Returns:
        List[ParameterValuePair]: list of all parameter-value-pairs supported by bashi
    """
    local_parameter_matrix = copy.deepcopy(parameter_matrix)

    def remove_host_compiler_nvcc(param_val: ParameterValue) -> bool:
        if param_val.name == NVCC:
            return False
        return True

    # remove nvcc as host compiler
    local_parameter_matrix[HOST_COMPILER] = list(
        filter(remove_host_compiler_nvcc, local_parameter_matrix[HOST_COMPILER])
    )

    # remove clang-cuda 13 and older
    def remove_unsupported_clang_cuda_version(param_val: ParameterValue) -> bool:
        if param_val.name == CLANG_CUDA and param_val.version < packaging.version.parse("14"):
            return False
        return True

    local_parameter_matrix[HOST_COMPILER] = list(
        filter(remove_unsupported_clang_cuda_version, local_parameter_matrix[HOST_COMPILER])
    )
    local_parameter_matrix[DEVICE_COMPILER] = list(
        filter(remove_unsupported_clang_cuda_version, local_parameter_matrix[DEVICE_COMPILER])
    )

    param_val_pair_list = get_expected_parameter_value_pairs(local_parameter_matrix)

    _remove_unsupported_nvcc_host_compiler(param_val_pair_list)
    _remove_different_compiler_names(param_val_pair_list)
    _remove_different_compiler_versions(param_val_pair_list)
    _remove_nvcc_unsupported_gcc_versions(param_val_pair_list)
    _remove_nvcc_unsupported_clang_versions(param_val_pair_list)
    _remove_specific_nvcc_clang_combinations(param_val_pair_list)
    _remove_unsupported_compiler_for_hip_backend(param_val_pair_list)
    _remove_disabled_hip_backend_for_hipcc(param_val_pair_list)
    _remove_enabled_sycl_backend_for_hipcc(param_val_pair_list)
    remove_parameter_value_pairs(
        param_val_pair_list,
        parameter1=ALPAKA_ACC_GPU_HIP_ENABLE,
        value_name1=ALPAKA_ACC_GPU_HIP_ENABLE,
        value_version1=ON,
        parameter2=ALPAKA_ACC_SYCL_ENABLE,
        value_name2=ALPAKA_ACC_SYCL_ENABLE,
        value_version2=ON,
    )
    _remove_enabled_cuda_backend_for_hipcc(param_val_pair_list)
    _remove_enabled_cuda_backend_for_enabled_hip_backend(param_val_pair_list)
    _remove_unsupported_compiler_for_sycl_backend(param_val_pair_list)
    _remove_disabled_sycl_backend_for_icpx(param_val_pair_list)
    _remove_enabled_hip_backend_for_icpx(param_val_pair_list)
    _remove_enabled_cuda_backend_for_icpx(param_val_pair_list)
    _remove_enabled_cuda_backend_for_enabled_sycl_backend(param_val_pair_list)
    _remove_nvcc_and_cuda_version_not_same(param_val_pair_list)
    _remove_cuda_sdk_unsupported_gcc_versions(param_val_pair_list)
    _remove_cuda_sdk_unsupported_clang_versions(param_val_pair_list)
    _remove_device_compiler_gcc_clang_enabled_cuda_backend(param_val_pair_list)
    _remove_specific_cuda_clang_combinations(param_val_pair_list)
    _remove_unsupported_clang_sdk_versions_for_clang_cuda(param_val_pair_list)

    return param_val_pair_list


def _remove_unsupported_nvcc_host_compiler(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all combinations where nvcc is device compiler and the host compiler is not gcc or
    clang.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    for compiler_name in set(COMPILERS) - set([GCC, CLANG, NVCC]):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            parameter1=HOST_COMPILER,
            value_name1=compiler_name,
            parameter2=DEVICE_COMPILER,
            value_name2=NVCC,
        )


def _remove_different_compiler_names(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all combinations, where host and device compiler name are different except the device
    compiler name is nvcc.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    # remove all combinations, where host and device compiler name are different except the device
    # compiler name is nvcc
    for host_compiler_name in set(COMPILERS) - set([NVCC]):
        for device_compiler_name in set(COMPILERS) - set([NVCC]):
            if host_compiler_name != device_compiler_name:
                remove_parameter_value_pairs(
                    parameter_value_pairs,
                    parameter1=HOST_COMPILER,
                    value_name1=host_compiler_name,
                    parameter2=DEVICE_COMPILER,
                    value_name2=device_compiler_name,
                )


def _remove_different_compiler_versions(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all combinations, where host and device compiler name are equal and versions are
    different except the compiler name is nvcc.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
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

    parameter_value_pairs[:] = list(filter(filter_function, parameter_value_pairs))


def _remove_nvcc_unsupported_gcc_versions(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all gcc version, which are to new for a specific nvcc version.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    _remove_unsupported_nvcc_cuda_host_compiler_versions(
        parameter_value_pairs, GCC, DEVICE_COMPILER, NVCC, NVCC_GCC_MAX_VERSION
    )


def _remove_nvcc_unsupported_clang_versions(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all clang version, which are to new for a specific nvcc version.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    _remove_unsupported_nvcc_cuda_host_compiler_versions(
        parameter_value_pairs, CLANG, DEVICE_COMPILER, NVCC, NVCC_CLANG_MAX_VERSION
    )


def _remove_unsupported_nvcc_cuda_host_compiler_versions(
    parameter_value_pairs: List[ParameterValuePair],
    host_compiler_name: str,
    second_parameter_name: Parameter,
    second_value_name: ValueName,
    support_list: List[NvccHostSupport],
):
    # pylint: disable=too-few-public-methods
    # pylint: disable=too-many-arguments
    class _FilterFunctor:
        def __init__(
            self,
            host_compiler_name: str,
            second_parameter_name: Parameter,
            second_value_name: ValueName,
            inklusiv_min_version: Optional[NvccHostSupport] = None,
            exklusiv_max_version: Optional[NvccHostSupport] = None,
        ) -> None:
            self.host_compiler_name = host_compiler_name
            self.second_parameter_name = second_parameter_name
            self.second_value_name = second_value_name
            if inklusiv_min_version and exklusiv_max_version:
                if inklusiv_min_version.host == exklusiv_max_version.host:
                    self.host_specifier_set = SpecifierSet(f">{exklusiv_max_version.host}")
                else:
                    self.host_specifier_set = SpecifierSet(f">={exklusiv_max_version.host}")
                self.nvcc_specifier_set = SpecifierSet(
                    f">={inklusiv_min_version.nvcc},<{exklusiv_max_version.nvcc}"
                )
            elif inklusiv_min_version:
                self.host_specifier_set = SpecifierSet(f">{inklusiv_min_version.host}")
                self.nvcc_specifier_set = SpecifierSet(f"=={inklusiv_min_version.nvcc}")
            elif exklusiv_max_version:
                self.host_specifier_set = SpecifierSet(f">{exklusiv_max_version.host}")
                self.nvcc_specifier_set = SpecifierSet(f"=={exklusiv_max_version.nvcc}")
            else:
                raise RuntimeError(
                    "at least inklusiv_min_version or exklusiv_max_version needs to be set"
                )

        def __call__(self, param_val_pair: ParameterValuePair) -> bool:
            if (
                param_val_pair.first.parameter == HOST_COMPILER
                and param_val_pair.second.parameter == self.second_parameter_name
            ):
                host_param_val = param_val_pair.first.parameterValue
                nvcc_param_val = param_val_pair.second.parameterValue
            elif (
                param_val_pair.first.parameter == self.second_parameter_name
                and param_val_pair.second.parameter == HOST_COMPILER
            ):
                host_param_val = param_val_pair.second.parameterValue
                nvcc_param_val = param_val_pair.first.parameterValue
            else:
                return True

            if (
                host_param_val.name == self.host_compiler_name
                and nvcc_param_val.name == self.second_value_name
            ):
                if (
                    nvcc_param_val.version in self.nvcc_specifier_set
                    and host_param_val.version in self.host_specifier_set
                ):
                    return False

            return True

    oldest_nvcc_first = sorted(support_list)

    for index in range(len(oldest_nvcc_first) - 1):
        filter_function = _FilterFunctor(
            host_compiler_name,
            second_parameter_name,
            second_value_name,
            oldest_nvcc_first[index],
            oldest_nvcc_first[index + 1],
        )

        parameter_value_pairs[:] = filter(filter_function, parameter_value_pairs)

    # lower bound
    parameter_value_pairs[:] = filter(
        _FilterFunctor(
            host_compiler_name,
            second_parameter_name,
            second_value_name,
            inklusiv_min_version=oldest_nvcc_first[0],
        ),
        parameter_value_pairs,
    )
    # upper bound
    parameter_value_pairs[:] = filter(
        _FilterFunctor(
            host_compiler_name,
            second_parameter_name,
            second_value_name,
            exklusiv_max_version=oldest_nvcc_first[-1],
        ),
        parameter_value_pairs,
    )


def _remove_specific_nvcc_clang_combinations(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all pairs, where clang is host-compiler for nvcc 11.3, 11.4 and 11.5 as device
    compiler.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    remove_parameter_value_pairs(
        parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=CLANG,
        value_version1=ANY_VERSION,
        parameter2=DEVICE_COMPILER,
        value_name2=NVCC,
        value_version2="!=11.3,!=11.4,!=11.5",
    )


def _remove_unsupported_compiler_for_hip_backend(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all pairs, where the hip backend is enabled and the compiler is not hipcc.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    for compiler_name in COMPILERS:
        if compiler_name != HIPCC:
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                remove_parameter_value_pairs(
                    parameter_value_pairs,
                    parameter1=compiler_type,
                    value_name1=compiler_name,
                    value_version1=ANY_VERSION,
                    parameter2=ALPAKA_ACC_GPU_HIP_ENABLE,
                    value_name2=ALPAKA_ACC_GPU_HIP_ENABLE,
                    value_version2=ON,
                )


def _remove_disabled_hip_backend_for_hipcc(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all pairs, where the hipcc is the compiler and the hip backend is disabled.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=HIPCC,
            value_version1=ANY_VERSION,
            parameter2=ALPAKA_ACC_GPU_HIP_ENABLE,
            value_name2=ALPAKA_ACC_GPU_HIP_ENABLE,
            value_version2=OFF,
        )


def _remove_enabled_sycl_backend_for_hipcc(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all pairs, where the hipcc is the compiler and the sycl backend is enabled.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=HIPCC,
            value_version1=ANY_VERSION,
            parameter2=ALPAKA_ACC_SYCL_ENABLE,
            value_name2=ALPAKA_ACC_SYCL_ENABLE,
            value_version2=ON,
        )


def _remove_enabled_cuda_backend_for_hipcc(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all pairs, where the hipcc is the compiler and the sycl backend is enabled.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=HIPCC,
            value_version1=ANY_VERSION,
            parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_name2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_version2="==0.0.0",
        )


def _remove_enabled_cuda_backend_for_enabled_hip_backend(
    parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where the hipcc is the compiler and the sycl backend is enabled.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    remove_parameter_value_pairs(
        parameter_value_pairs,
        parameter1=ALPAKA_ACC_GPU_HIP_ENABLE,
        value_name1=ALPAKA_ACC_GPU_HIP_ENABLE,
        value_version1=ON,
        parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_name2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_version2="==0.0.0",
    )


def _remove_unsupported_compiler_for_sycl_backend(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all pairs, where the hip backend is enabled and the compiler is not hipcc.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    for compiler_name in COMPILERS:
        if compiler_name != ICPX:
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                remove_parameter_value_pairs(
                    parameter_value_pairs,
                    parameter1=compiler_type,
                    value_name1=compiler_name,
                    value_version1=ANY_VERSION,
                    parameter2=ALPAKA_ACC_SYCL_ENABLE,
                    value_name2=ALPAKA_ACC_SYCL_ENABLE,
                    value_version2=ON,
                )


def _remove_disabled_sycl_backend_for_icpx(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all pairs, where the hipcc is the compiler and the hip backend is disabled.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=ICPX,
            value_version1=ANY_VERSION,
            parameter2=ALPAKA_ACC_SYCL_ENABLE,
            value_name2=ALPAKA_ACC_SYCL_ENABLE,
            value_version2=OFF,
        )


def _remove_enabled_hip_backend_for_icpx(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all pairs, where the hipcc is the compiler and the sycl backend is enabled.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=ICPX,
            value_version1=ANY_VERSION,
            parameter2=ALPAKA_ACC_GPU_HIP_ENABLE,
            value_name2=ALPAKA_ACC_GPU_HIP_ENABLE,
            value_version2=ON,
        )


def _remove_enabled_cuda_backend_for_icpx(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all pairs, where the hipcc is the compiler and the sycl backend is enabled.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=ICPX,
            value_version1=ANY_VERSION,
            parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_name2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_version2="==0.0.0",
        )


def _remove_enabled_cuda_backend_for_enabled_sycl_backend(
    parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where the hipcc is the compiler and the sycl backend is enabled.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    remove_parameter_value_pairs(
        parameter_value_pairs,
        parameter1=ALPAKA_ACC_SYCL_ENABLE,
        value_name1=ALPAKA_ACC_SYCL_ENABLE,
        value_version1=ON,
        parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_name2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_version2="==0.0.0",
    )


def _remove_nvcc_and_cuda_version_not_same(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all pairs, where the device compiler version of nvcc is not equal to the CUDA backend.
    Filters also the disabled backend, because there is no nvcc@OFF.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """

    def filter_function(param_val_pair: ParameterValuePair) -> bool:
        for param_val1, param_val2 in (
            (param_val_pair.first, param_val_pair.second),
            (param_val_pair.second, param_val_pair.first),
        ):
            if (
                param_val1.parameter == DEVICE_COMPILER
                and param_val1.parameterValue.name == NVCC
                and param_val2.parameter == ALPAKA_ACC_GPU_CUDA_ENABLE
                and param_val1.parameterValue.version != param_val2.parameterValue.version
            ):
                return False

        return True

    parameter_value_pairs[:] = list(filter(filter_function, parameter_value_pairs))


def _remove_cuda_sdk_unsupported_gcc_versions(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all gcc version, which are to new for a specific cuda sdk version.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    _remove_unsupported_nvcc_cuda_host_compiler_versions(
        parameter_value_pairs,
        GCC,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        NVCC_GCC_MAX_VERSION,
    )


def _remove_cuda_sdk_unsupported_clang_versions(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all clang version, which are to new for a specific cuda sdk version.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    _remove_unsupported_nvcc_cuda_host_compiler_versions(
        parameter_value_pairs,
        CLANG,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        NVCC_CLANG_MAX_VERSION,
    )


def _remove_device_compiler_gcc_clang_enabled_cuda_backend(
    parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs where clang or gcc is device compiler the CUDA backend is enabled.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    for compiler in (GCC, CLANG):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            parameter1=DEVICE_COMPILER,
            value_name1=compiler,
            value_version1=ANY_VERSION,
            parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_name2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_version2="==0.0.0",
        )


def _remove_specific_cuda_clang_combinations(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all pairs, where clang is host-compiler for cuda sdk 11.3, 11.4 and 11.5.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    remove_parameter_value_pairs(
        parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=CLANG,
        value_version1=ANY_VERSION,
        parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_name2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_version2="!=11.3,!=11.4,!=11.5",
    )


def _remove_unsupported_clang_sdk_versions_for_clang_cuda(
    parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all CUDA SDK versions, which are not supported by a specific clang-cuda version.
    Includes also disabled CUDA backends.

    If clang-cuda version is unknown and older than the oldest supported clang-cuda version, filter
    out all versions which are not supported by the oldest supported clang-cuda version.

    If clang-cuda version is new, than the latest supported clang-cuda version, do not filter it.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """

    def filter_func(param_val_pair: ParameterValuePair) -> bool:
        # pylint: disable=too-many-nested-blocks
        for param_val1, param_val2 in (
            (param_val_pair.first, param_val_pair.second),
            (param_val_pair.second, param_val_pair.first),
        ):
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                if (
                    param_val1.parameter == compiler_type
                    and param_val1.parameterValue.name == CLANG_CUDA
                    and param_val2.parameter == ALPAKA_ACC_GPU_CUDA_ENABLE
                ):
                    if param_val2.parameterValue.version == OFF_VER:
                        return False

                    # if clang-cuda is newer than the latest supported clang-cuda version, we needs
                    # to assume, that it supports every CUDA SDK version
                    if (
                        param_val1.parameterValue.version
                        > CLANG_CUDA_MAX_CUDA_VERSION[0].clang_cuda
                    ):
                        return True

                    # create variable, that it is not unbound in the else branch of the for loop
                    # create dummy object to avoid None
                    version_combination = ClangCudaSDKSupport("0", "9999")
                    for version_combination in CLANG_CUDA_MAX_CUDA_VERSION:
                        if param_val1.parameterValue.version >= version_combination.clang_cuda:
                            if param_val2.parameterValue.version > version_combination.cuda:
                                return False
                            break
                    # if clang-cuda versions is older than the last supported clang-cuda version
                    # every CUDA SDK version can pass, which is not forbidden by the oldest
                    # supported clang-cuda version
                    else:
                        if param_val1.parameterValue.version < version_combination.clang_cuda:
                            if param_val2.parameterValue.version > version_combination.cuda:
                                return False
        return True

    parameter_value_pairs[:] = filter(filter_func, parameter_value_pairs)


def _remove_unsupported_gcc_versions_for_ubuntu2004(
    parameter_value_pairs: List[ParameterValuePair],
):
    """Remove pairs where GCC version 6 is used with Ubuntu 20.04.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=GCC,
            value_version1="!=6",
            parameter2=UBUNTU,
            value_name2=UBUNTU,
            value_version2="!=20.04",
        )
