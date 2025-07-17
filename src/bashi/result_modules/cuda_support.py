"""Filter rules to remove combinations which has to do with the CUDA"""

from typing import List, Optional, Dict, Callable
from packaging.specifiers import SpecifierSet
from bashi.types import ParameterValueSingle, ParameterValuePair
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import (
    COMPILERS,
    NVCC_GCC_MAX_VERSION,
    NVCC_CLANG_MAX_VERSION,
    CLANG_CUDA_MAX_CUDA_VERSION,
    UBUNTU_CUDA_VERSION_RANGE,
    UBUNTU_CLANG_CUDA_SDK_SUPPORT,
    NvccHostSupport,
    ClangCudaSDKSupport,
)
from bashi.utils import remove_parameter_value_pairs_ranges, bi_filter
from bashi.result_modules.sdk_helper import (
    remove_unsupported_sdk_ubuntu_combinations,
    remove_runtime_not_available_ubuntu_versions,
)


def remove_cuda_specific_parameter_value_pairs(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    runtime_infos: Dict[str, Callable[..., bool]],
):
    """Apply several filter functions to remove invalid CUDA related parameter-value-pairs.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
        removed_parameter_value_pairs (List[ParameterValuePair): list with removed
            parameter-value-pairs
        runtime_infos: Dict[str, Callable[..., bool]]: Dict of functors which checks if the given
        parameter-values (combinations) are valid.
    """
    _remove_nvcc_host_compiler(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_unsupported_clang_cuda_version(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_unsupported_nvcc_host_compiler(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_nvcc_unsupported_gcc_versions(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_nvcc_unsupported_clang_versions(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_specific_nvcc_clang_combinations(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_nvcc_and_cuda_version_not_same(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_cuda_sdk_unsupported_gcc_versions(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_cuda_sdk_unsupported_clang_versions(
        parameter_value_pairs, removed_parameter_value_pairs
    )
    _remove_device_compiler_gcc_clang_enabled_cuda_backend(
        parameter_value_pairs, removed_parameter_value_pairs
    )
    _remove_specific_cuda_clang_combinations(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_unsupported_clang_sdk_versions_for_clang_cuda(
        parameter_value_pairs, removed_parameter_value_pairs
    )
    _remove_unsupported_cmake_versions_for_clangcuda(
        parameter_value_pairs, removed_parameter_value_pairs
    )
    _remove_unsupported_nvcc_ubuntu_combinations(
        parameter_value_pairs, removed_parameter_value_pairs
    )
    _remove_unsupported_cuda_backend_ubuntu_combinations(
        parameter_value_pairs, removed_parameter_value_pairs
    )
    _remove_unsupported_clang_cuda_ubuntu_combinations(
        parameter_value_pairs, removed_parameter_value_pairs
    )
    _remove_runtime_unsupported_cuda_backend_ubuntu_combinations(
        parameter_value_pairs, removed_parameter_value_pairs, runtime_infos
    )
    _remove_runtime_unsupported_clang_cuda_ubuntu_combinations(
        parameter_value_pairs, removed_parameter_value_pairs, runtime_infos
    )


def _remove_nvcc_host_compiler(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove nvcc as host compiler.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    remove_parameter_value_pairs_ranges(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=NVCC,
    )


def _remove_unsupported_clang_cuda_version(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove Clang-CUDA 13 and older

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=CLANG_CUDA,
            value_max_version1=13,
        )


def _remove_unsupported_nvcc_host_compiler(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all combinations where nvcc is device compiler and the host compiler is not gcc or
    clang.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_name in set(COMPILERS) - set([GCC, CLANG, NVCC]):
        remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=HOST_COMPILER,
            value_name1=compiler_name,
            parameter2=DEVICE_COMPILER,
            value_name2=NVCC,
        )


def _remove_nvcc_unsupported_gcc_versions(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all gcc version, which are to new for a specific nvcc version.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    _remove_unsupported_nvcc_cuda_host_compiler_versions(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        GCC,
        DEVICE_COMPILER,
        NVCC,
        NVCC_GCC_MAX_VERSION,
    )


def _remove_nvcc_unsupported_clang_versions(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all clang version, which are to new for a specific nvcc version.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    _remove_unsupported_nvcc_cuda_host_compiler_versions(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        CLANG,
        DEVICE_COMPILER,
        NVCC,
        NVCC_CLANG_MAX_VERSION,
    )


# pylint: disable=too-many-positional-arguments
def _remove_unsupported_nvcc_cuda_host_compiler_versions(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    host_compiler_name: str,
    second_parameter_name: Parameter,
    second_value_name: ValueName,
    support_list: List[NvccHostSupport],
):
    # pylint: disable=too-few-public-methods
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
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

        bi_filter(parameter_value_pairs, removed_parameter_value_pairs, filter_function)

    # lower bound
    bi_filter(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        _FilterFunctor(
            host_compiler_name,
            second_parameter_name,
            second_value_name,
            inklusiv_min_version=oldest_nvcc_first[0],
        ),
    )
    # upper bound
    bi_filter(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        _FilterFunctor(
            host_compiler_name,
            second_parameter_name,
            second_value_name,
            exklusiv_max_version=oldest_nvcc_first[-1],
        ),
    )


def _remove_specific_nvcc_clang_combinations(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where clang is host-compiler for nvcc 11.3, 11.4 and 11.5 as device
    compiler.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    remove_parameter_value_pairs_ranges(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=CLANG,
        parameter2=DEVICE_COMPILER,
        value_name2=NVCC,
        value_min_version2="11.3",
        value_max_version2="11.5",
    )


def _remove_nvcc_and_cuda_version_not_same(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where the device compiler version of nvcc is not equal to the CUDA backend.
    Filters also the disabled backend, because there is no nvcc@OFF.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
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

    bi_filter(parameter_value_pairs, removed_parameter_value_pairs, filter_function)


def _remove_cuda_sdk_unsupported_gcc_versions(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all gcc version, which are to new for a specific cuda sdk version.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    _remove_unsupported_nvcc_cuda_host_compiler_versions(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        GCC,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        NVCC_GCC_MAX_VERSION,
    )


def _remove_cuda_sdk_unsupported_clang_versions(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all clang version, which are to new for a specific cuda sdk version.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    _remove_unsupported_nvcc_cuda_host_compiler_versions(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        CLANG,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        NVCC_CLANG_MAX_VERSION,
    )


def _remove_device_compiler_gcc_clang_enabled_cuda_backend(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs where clang or gcc is device compiler the CUDA backend is enabled.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler in (GCC, CLANG):
        remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=DEVICE_COMPILER,
            value_name1=compiler,
            parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_name2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_min_version2=OFF,
            value_min_version2_inclusive=False,
        )


def _remove_specific_cuda_clang_combinations(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs, where clang is host-compiler for cuda sdk 11.3, 11.4 and 11.5.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    remove_parameter_value_pairs_ranges(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=CLANG,
        parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_name2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_min_version2=11.3,
        value_max_version2=11.5,
    )


def _remove_unsupported_clang_sdk_versions_for_clang_cuda(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all CUDA SDK versions, which are not supported by a specific clang-cuda version.
    Includes also disabled CUDA backends.

    If clang-cuda version is unknown and older than the oldest supported clang-cuda version, filter
    out all versions which are not supported by the oldest supported clang-cuda version.

    If clang-cuda version is new, than the latest supported clang-cuda version, do not filter it.

    parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
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

    bi_filter(parameter_value_pairs, removed_parameter_value_pairs, filter_func)


def _remove_unsupported_cmake_versions_for_clangcuda(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove CMake 3.18 if Clang-CUDA is the host or device compiler.

    Args:

    parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=CLANG_CUDA,
            parameter2=CMAKE,
            value_name2=CMAKE,
            value_max_version2="3.18",
        )


def _remove_unsupported_nvcc_ubuntu_combinations(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs where NVCC does not support a specific Ubuntu version

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
        removed_parameter_value_pairs (List[ParameterValuePair): list with removed
            parameter-value-pairs
    """
    remove_unsupported_sdk_ubuntu_combinations(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        DEVICE_COMPILER,
        NVCC,
        UBUNTU_CUDA_VERSION_RANGE,
    )


def _remove_unsupported_cuda_backend_ubuntu_combinations(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs where the CUDA backend does not support a specific Ubuntu version

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
        removed_parameter_value_pairs (List[ParameterValuePair): list with removed
            parameter-value-pairs
    """
    remove_unsupported_sdk_ubuntu_combinations(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        UBUNTU_CUDA_VERSION_RANGE,
    )


def _remove_unsupported_clang_cuda_ubuntu_combinations(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all pairs where Clang-CUDA does not support a specific Ubuntu version

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
            CLANG_CUDA,
            UBUNTU_CLANG_CUDA_SDK_SUPPORT,
        )


def _remove_runtime_unsupported_cuda_backend_ubuntu_combinations(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    runtime_infos: Dict[str, Callable[..., bool]],
):
    """Remove all pairs Ubuntu CUDA backend pairs, which are not support for the given runtime info.
    Depending on the given CUDA versions in the input parameter-value-matrix it is possible that
    there is no CUDA version which can be installed a specific Ubuntu version.

    For example CUDA 11.0 until 11.9 can be installed on Ubuntu 20.04. If no HIPCC 11.0 - 11.9 is
    in the input matrix, there is no possibility to install HIP on Ubuntu 20.04

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
        removed_parameter_value_pairs (List[ParameterValuePair): list with removed
            parameter-value-pairs
        runtime_infos: Dict[str, Callable[..., bool]]: Dict of functors which checks if the given
        parameter-values (combinations) are valid. This filter uses the functor
            RT_AVAILABLE_CUDA_SDK_UBUNTU_VER.
    """
    if RT_AVAILABLE_CUDA_SDK_UBUNTU_VER not in runtime_infos:
        return

    remove_runtime_not_available_ubuntu_versions(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        runtime_infos[RT_AVAILABLE_CUDA_SDK_UBUNTU_VER],
        ALPAKA_ACC_GPU_CUDA_ENABLE,
    )


def _remove_runtime_unsupported_clang_cuda_ubuntu_combinations(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    runtime_infos: Dict[str, Callable[..., bool]],
):
    """Remove all pairs Ubuntu Clang-CUDA backend pairs, which are not support for the given runtime
    info. Depending on the given Clang-CUDA versions in the input parameter-value-matrix it is
    possible that there is no CUDA version which can be installed a specific Ubuntu version.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
        removed_parameter_value_pairs (List[ParameterValuePair): list with removed
            parameter-value-pairs
        runtime_infos: Dict[str, Callable[..., bool]]: Dict of functors which checks if the given
        parameter-values (combinations) are valid. This filter uses the functor
            RT_AVAILABLE_CUDA_SDK_UBUNTU_VER.
    """
    if RT_AVAILABLE_CUDA_SDK_UBUNTU_VER not in runtime_infos:
        return

    tmp_parameter_value_pairs: List[ParameterValuePair] = []

    def is_remove(param_val1: ParameterValueSingle, param_val2: ParameterValueSingle) -> bool:
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            if (
                param_val1.parameter == compiler_type
                and param_val1.parameterValue.name == CLANG_CUDA
                and param_val2.parameter == UBUNTU
                and not runtime_infos[RT_AVAILABLE_CUDA_SDK_UBUNTU_VER](
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
