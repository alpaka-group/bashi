"""Filter rules basing on host and device compiler names and versions.

All rules implemented in this filter have an identifier that begins with "v" and follows a number. 
Examples: v1, v42, v678 ...

These identifiers are used in the test names, for example, to make it clear which test is testing 
which rule.
"""

from typing import Optional, IO, List
from typeguard import typechecked
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import Parameter, ParameterValueTuple
from bashi.versions import NVCC_GCC_MAX_VERSION, NVCC_CLANG_MAX_VERSION
from bashi.utils import reason


def get_required_parameters() -> List[Parameter]:
    """Return list of parameters which will be checked in the filter.

    Returns:
        List[Parameter]: list of checked parameters
    """
    return [HOST_COMPILER, DEVICE_COMPILER]


@typechecked
def compiler_version_filter_typechecked(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Type-checked version of compiler_version_filter(). Type checking has a big performance cost,
    which is why the non type-checked version is used for the pairwise generator.
    """
    return compiler_version_filter(row, output)


def compiler_version_filter(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Filter rules basing on host and device compiler names and versions.

    Args:
        row (ParameterValueTuple): parameter-value-tuple to verify.
        output (Optional[IO[str]], optional): Writes the reason in the io object why the parameter
            value tuple does not pass the filter. If None, no information is provided. The default
            value is None.

    Returns:
        bool: True, if parameter-value-tuple is valid.
    """

    # Rule: v1
    if (
        DEVICE_COMPILER in row
        and row[DEVICE_COMPILER].name != NVCC
        and HOST_COMPILER in row
        and row[HOST_COMPILER].version != row[DEVICE_COMPILER].version
    ):
        reason(output, "host and device compiler version must be the same (except for nvcc)")
        return False

    # now idea, how remove nested blocks without hitting the performance
    # pylint: disable=too-many-nested-blocks
    if DEVICE_COMPILER in row and row[DEVICE_COMPILER].name == NVCC:
        if HOST_COMPILER in row and row[HOST_COMPILER].name == GCC:
            # Rule: v2
            # remove all unsupported nvcc gcc version combinations
            # define which is the latest supported gcc compiler for a nvcc version

            # if a nvcc version is not supported by bashi, assume that the version supports the
            # latest gcc compiler version
            if row[DEVICE_COMPILER].version <= NVCC_GCC_MAX_VERSION[0].nvcc:
                # check the maximum supported gcc version for the given nvcc version
                for nvcc_gcc_comb in NVCC_GCC_MAX_VERSION:
                    if row[DEVICE_COMPILER].version >= nvcc_gcc_comb.nvcc:
                        if row[HOST_COMPILER].version > nvcc_gcc_comb.host:
                            reason(
                                output,
                                f"nvcc {row[DEVICE_COMPILER].version} "
                                f"does not support gcc {row[HOST_COMPILER].version}",
                            )
                            return False
                        break

        if HOST_COMPILER in row and row[HOST_COMPILER].name == CLANG:
            # Rule: v3
            # remove all unsupported nvcc clang version combinations
            # define which is the latest supported clang compiler for a nvcc version

            # if a nvcc version is not supported by bashi, assume that the version supports the
            # latest clang compiler version
            if row[DEVICE_COMPILER].version <= NVCC_CLANG_MAX_VERSION[0].nvcc:
                # check the maximum supported gcc version for the given nvcc version
                for nvcc_clang_comb in NVCC_CLANG_MAX_VERSION:
                    if row[DEVICE_COMPILER].version >= nvcc_clang_comb.nvcc:
                        if row[HOST_COMPILER].version > nvcc_clang_comb.host:
                            reason(
                                output,
                                f"nvcc {row[DEVICE_COMPILER].version} "
                                f"does not support clang {row[HOST_COMPILER].version}",
                            )
                            return False
                        break

    return True
