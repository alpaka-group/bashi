"""Filter rules to remove compiler C++ standard combination, which are not supported."""

from typing import List
from bashi.types import ParameterValuePair
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import (
    CompilerCxxSupport,
    GCC_CXX_SUPPORT_VERSION,
    CLANG_CXX_SUPPORT,
    NVCC_CXX_SUPPORT_VERSION,
)
from bashi.utils import remove_parameter_value_pairs_ranges


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
            CLANG_CXX_SUPPORT,
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
