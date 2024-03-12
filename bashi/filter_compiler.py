"""Filter rules basing on host and device compiler names and versions.

All rules implemented in this filter have an identifier that begins with "c" and follows a number. 
Examples: c1, c42, c678 ...

These identifiers are used in the test names, for example, to make it clear which test is testing 
which rule.
"""

from typing import Optional, IO, List
import packaging.version as pkv
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
def compiler_filter_typechecked(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Type-checked version of compiler_filter(). Type checking has a big performance cost, which
    is why the non type-checked version is used for the pairwise generator.
    """
    return compiler_filter(row, output)


# pylint: disable=too-many-branches
# pylint: disable=too-many-return-statements
def compiler_filter(
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
    # Rule: c1
    # NVCC as HOST_COMPILER is not allow
    # this rule will be never used, because of an implementation detail of the covertable library
    # it is not possible to add NVCC as HOST_COMPILER and filter out afterwards
    # this rule is only used by bashi-verify
    if HOST_COMPILER in row and row[HOST_COMPILER].name == NVCC:
        reason(output, "nvcc is not allowed as host compiler")
        return False

    # Rule: c2
    if (
        DEVICE_COMPILER in row
        and row[DEVICE_COMPILER].name == NVCC
        and HOST_COMPILER in row
        and not row[HOST_COMPILER].name in [GCC, CLANG]
    ):
        reason(output, "only gcc and clang are allowed as nvcc host compiler")
        return False

    # Rule: c3
    if (
        DEVICE_COMPILER in row
        and row[DEVICE_COMPILER].name != NVCC
        and HOST_COMPILER in row
        and row[HOST_COMPILER].name != row[DEVICE_COMPILER].name
    ):
        reason(output, "host and device compiler name must be the same (except for nvcc)")
        return False

    # Rule: c4
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
            # Rule: c5
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
            # Rule: c7
            if row[DEVICE_COMPILER].version >= pkv.parse("11.3") and row[
                DEVICE_COMPILER
            ].version <= pkv.parse("11.5"):
                reason(
                    output,
                    "clang as host compiler is disabled for nvcc 11.3 to 11.5",
                )
                return False

            # Rule: c6
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

    # Rule: c8
    # clang-cuda 13 and older is not supported
    # this rule will be never used, because of an implementation detail of the covertable library
    # it is not possible to add the clang-cuda versions and filter it out afterwards
    # this rule is only used by bashi-verify
    if (
        DEVICE_COMPILER in row
        and row[DEVICE_COMPILER].name == CLANG_CUDA
        and row[DEVICE_COMPILER].version < pkv.parse("14")
    ):
        reason(output, "all clang versions older than 14 are disabled as CUDA Compiler")
        return False

    return True
