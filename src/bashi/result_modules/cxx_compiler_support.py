"""Filter rules to remove compiler C++ standard combination, which are not supported."""

from typing import List, Dict
from dataclasses import dataclass
import packaging.version as pkv
from bashi.types import ParameterValuePair
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import (
    CompilerCxxSupport,
    GCC_CXX_SUPPORT_VERSION,
    CLANG_CXX_SUPPORT_VERSION,
    NVCC_CXX_SUPPORT_VERSION,
    CLANG_CUDA_CXX_SUPPORT_VERSION,
    MAX_CUDA_SDK_CXX_SUPPORT,
    ICPX_CXX_SUPPORT_VERSION,
    HIPCC_CXX_SUPPORT_VERSION,
)
from bashi.utils import remove_parameter_value_pairs_ranges


def remove_cxx_specific_parameter_value_pairs(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Apply several filter functions to remove C++ standard related parameter-value-pairs.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
        removed_parameter_value_pairs (List[ParameterValuePair): list with removed
            parameter-value-pairs
    """
    _remove_unsupported_cxx_versions_for_gcc(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_unsupported_cxx_versions_for_clang(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_unsupported_cxx_versions_for_nvcc(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_unsupported_cxx_versions_for_clang_cuda(
        parameter_value_pairs, removed_parameter_value_pairs
    )
    _remove_unsupported_cxx_versions_for_cuda(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_unsupported_cxx_versions_for_icpx(parameter_value_pairs, removed_parameter_value_pairs)
    _remove_unsupported_cxx_versions_for_hipcc(parameter_value_pairs, removed_parameter_value_pairs)


def _remove_unsupported_cxx_version_for_compiler(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    compiler_name: str,
    compiler_cxx_support_list: List[CompilerCxxSupport],
    compiler_type: str,
):
    """Remove unsupported combinations of compiler versions and C++ standard.

    Args:

    parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    compiler_name (str): name of the compiler
    compiler_cxx_support_list List[CompilerCxxSupport]: list containing which compiler version added
        support for a new C++ standard
    compiler_type: HOST_COMPILER or DEVICE_COMPILER
    """
    sorted_compiler_cxx_supported_version = sorted(compiler_cxx_support_list)
    # loop over version ranges
    # first iteration: handle all GCC version older then the oldest defined version
    # n+1 iterations: handle versions between the defined supported GCC versions
    # last iteration: handle all GCC versions younger than the latest defined GCC version
    compiler_cxx_ver = sorted_compiler_cxx_supported_version[0]
    compiler_min_ver: str = ANY_VERSION
    cxx_min_ver: int = int(str(compiler_cxx_ver.cxx)) - 3

    if cxx_min_ver < 11:
        raise RuntimeError("Does not support minium C++ version older than 14.")

    for i in range(len(sorted_compiler_cxx_supported_version) + 1):
        if i < len(sorted_compiler_cxx_supported_version):
            compiler_cxx_ver = sorted_compiler_cxx_supported_version[i]
            compiler_max_ver: str = str(compiler_cxx_ver.compiler)
        else:
            compiler_max_ver = ANY_VERSION

        remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=compiler_type,
            value_name1=compiler_name,
            value_min_version1=compiler_min_ver,
            value_max_version1=compiler_max_ver,
            value_min_version1_inclusive=True,
            value_max_version1_inclusive=False,
            parameter2=CXX_STANDARD,
            value_min_version2=cxx_min_ver,
            value_min_version2_inclusive=False,
            value_max_version2_inclusive=False,
        )

        if i < len(sorted_compiler_cxx_supported_version):
            compiler_min_ver = str(compiler_cxx_ver.compiler)
            cxx_min_ver = int(str(compiler_cxx_ver.cxx))


def _remove_unsupported_cxx_versions_for_gcc(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove unsupported combinations of GCC compiler versions and C++ standard.

    Args:

    parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        _remove_unsupported_cxx_version_for_compiler(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            GCC,
            GCC_CXX_SUPPORT_VERSION,
            compiler_type,
        )


def _remove_unsupported_cxx_versions_for_clang(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove unsupported combinations of Clang compiler versions and C++ standard.

    Args:

    parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        _remove_unsupported_cxx_version_for_compiler(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            CLANG,
            CLANG_CXX_SUPPORT_VERSION,
            compiler_type,
        )


def _remove_unsupported_cxx_versions_for_nvcc(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove unsupported combinations of Nvcc compiler versions and C++ standard.

    Args:

    parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    _remove_unsupported_cxx_version_for_compiler(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        NVCC,
        NVCC_CXX_SUPPORT_VERSION,
        DEVICE_COMPILER,
    )


def _remove_unsupported_cxx_versions_for_clang_cuda(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove unsupported combinations of Clang-CUDA compiler versions and C++ standard.

    Args:

    parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        _remove_unsupported_cxx_version_for_compiler(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            CLANG_CUDA,
            CLANG_CUDA_CXX_SUPPORT_VERSION,
            compiler_type,
        )


def _remove_unsupported_cxx_versions_for_cuda(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove all combinations of the CUDA backend and the C++ standard, which are not possible with
    the Nvcc or Clang-CUDA compiler.

    Args:

    parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """

    @dataclass
    class CUDASdkRange:
        """Stores a version range, where is a C++ standard supported"""

        min: str
        max: str

    cxx_bounds: Dict[str, CUDASdkRange] = {}

    max_cuda_sdk_cxx_support_sorted = sorted(MAX_CUDA_SDK_CXX_SUPPORT)
    # because a Clang-CUDA version supports the latest CUDA SDK automatically, a upper bound is set
    for max_clang_cuda_sdk in max_cuda_sdk_cxx_support_sorted:
        cxx_bounds[str(max_clang_cuda_sdk.cxx)] = CUDASdkRange(
            str(max_clang_cuda_sdk.compiler), str(max_cuda_sdk_cxx_support_sorted[-1].compiler)
        )

    for max_nvcc_sdk_support in NVCC_CXX_SUPPORT_VERSION:
        cxx = str(max_nvcc_sdk_support.cxx)
        # check who supports a C++ standard earlier
        # if a C++ standard is supported by Nvcc all newer version also supports the C++
        # if a C++ standard is only supported by Clang-CUDA, there is a upper bound for the CUDA Sdk
        if cxx in cxx_bounds:
            cxx_bounds[cxx].min = min(cxx_bounds[cxx].min, str(max_nvcc_sdk_support.compiler))
            cxx_bounds[cxx].max = ANY_VERSION
        else:
            cxx_bounds[cxx] = CUDASdkRange(str(max_nvcc_sdk_support.compiler), ANY_VERSION)

    # sort the dict from the oldest to the latest C++ version
    cxx_bounds = dict(sorted(cxx_bounds.items()))

    # handle special case, if older C++ standard has the same or greater CUDA SDK version than the
    # successor C++ standard
    # Real world example:
    # - Since Nvcc 10.0 C++14 is supported
    # - Since Clang-CUDA 8, which supports C++17, CUDA 10.0 is supported
    # Remove C++ standard if it is "overlaid" by it's successor
    keys = list(cxx_bounds.keys())
    for i in range(len(keys) - 1):
        if pkv.parse(cxx_bounds[keys[i]].min) >= pkv.parse(cxx_bounds[keys[i + 1]].min):
            del cxx_bounds[keys[i]]

    for cxx, cuda_sdk_range in cxx_bounds.items():
        remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_name1=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_min_version1=OFF,
            value_min_version1_inclusive=False,
            value_max_version1=cuda_sdk_range.min,
            value_max_version1_inclusive=False,
            parameter2=CXX_STANDARD,
            value_name2=CXX_STANDARD,
            value_min_version2=cxx,
            value_min_version2_inclusive=True,
            value_max_version2=cxx,
            value_max_version2_inclusive=True,
        )
        if cuda_sdk_range.max != ANY_VERSION:
            remove_parameter_value_pairs_ranges(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                parameter1=ALPAKA_ACC_GPU_CUDA_ENABLE,
                value_name1=ALPAKA_ACC_GPU_CUDA_ENABLE,
                value_min_version1=cuda_sdk_range.max,
                value_min_version1_inclusive=False,
                parameter2=CXX_STANDARD,
                value_name2=CXX_STANDARD,
                value_min_version2=cxx,
                value_min_version2_inclusive=True,
                value_max_version2=cxx,
                value_max_version2_inclusive=True,
            )

    remove_parameter_value_pairs_ranges(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_name1=ALPAKA_ACC_GPU_CUDA_ENABLE,
        parameter2=CXX_STANDARD,
        value_name2=CXX_STANDARD,
        value_min_version2=list(cxx_bounds.keys())[-1],
        value_min_version2_inclusive=False,
    )


def _remove_unsupported_cxx_versions_for_icpx(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove unsupported combinations of ICPX compiler versions and C++ standard.

    Args:

    parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        _remove_unsupported_cxx_version_for_compiler(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            ICPX,
            ICPX_CXX_SUPPORT_VERSION,
            compiler_type,
        )


def _remove_unsupported_cxx_versions_for_hipcc(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove unsupported combinations of HIPCC compiler versions and C++ standard.

    Args:

    parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        _remove_unsupported_cxx_version_for_compiler(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            HIPCC,
            HIPCC_CXX_SUPPORT_VERSION,
            compiler_type,
        )
