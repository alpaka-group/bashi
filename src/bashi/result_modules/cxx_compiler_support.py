"""Filter rules to remove compiler C++ standard combination, which are not supported."""

from typing import List
from bashi.types import ParameterValuePair
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import GCC_CXX_SUPPORT_VERSION
from bashi.utils import remove_parameter_value_pairs_ranges


def _remove_unsupported_cxx_versions_for_gcc(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
):
    """Remove unsupported combinations of GCC compiler versions and C++ standard.

    Args:

    parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
    removed_parameter_value_pairs (List[ParameterValuePair): list with removed parameter-value-pairs
    """
    sorted_gcc_cxx_supported_version = sorted(GCC_CXX_SUPPORT_VERSION)
    # loop over version ranges
    # first iteration: handle all GCC version older then the oldest defined version
    # n+1 iterations: handle versions between the defined supported GCC versions
    # last iteration: handle all GCC versions younger than the latest defined GCC version
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        gcc_cxx_ver = sorted_gcc_cxx_supported_version[0]
        gcc_min_ver: str = ANY_VERSION
        cxx_min_ver: int = int(str(gcc_cxx_ver.cxx)) - 3

        if cxx_min_ver < 11:
            raise RuntimeError("Does not support minium C++ version older than 11.")

        for i in range(len(sorted_gcc_cxx_supported_version) + 1):
            if i < len(sorted_gcc_cxx_supported_version):
                gcc_cxx_ver = sorted_gcc_cxx_supported_version[i]
                gcc_max_ver: str = str(gcc_cxx_ver.gcc)
            else:
                gcc_max_ver = ANY_VERSION

            remove_parameter_value_pairs_ranges(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                parameter1=compiler_type,
                value_name1=GCC,
                value_min_version1=gcc_min_ver,
                value_max_version1=gcc_max_ver,
                value_min_version1_inclusive=True,
                value_max_version1_inclusive=False,
                parameter2=CXX_STANDARD,
                value_min_version2=cxx_min_ver,
                value_min_version2_inclusive=False,
                value_max_version2_inclusive=False,
            )

            if i < len(sorted_gcc_cxx_supported_version):
                gcc_min_ver = str(gcc_cxx_ver.gcc)
                cxx_min_ver = int(str(gcc_cxx_ver.cxx))
