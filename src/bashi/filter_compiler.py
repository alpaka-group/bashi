"""Filter rules basing on host and device compiler names and versions.

All rules implemented in this filter have an identifier that begins with "c" and follows a number.
Examples: c1, c42, c678 ...

These identifiers are used in the test names, for example, to make it clear which test is testing
which rule.
"""

from typing import Optional, IO
import packaging.version as pkv
from typeguard import typechecked
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValueTuple
from bashi.versions import (
    NVCC_GCC_MAX_VERSION,
    NVCC_CLANG_MAX_VERSION,
    CLANG_CUDA_MAX_CUDA_VERSION,
    GCC_CXX_SUPPORT_VERSION,
)
from bashi.utils import reason

# uncomment me for debugging
# from bashi.utils import print_row_nice


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
# pylint: disable=too-many-statements
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
    # uncomment me for debugging
    # print_row_nice(row, bashi_validate=False)

    # Rule: c1
    # NVCC as HOST_COMPILER is not allow
    # this rule will be never used, because of an implementation detail of the covertable library
    # it is not possible to add NVCC as HOST_COMPILER and filter out afterwards
    # this rule is only used by bashi-verify
    if HOST_COMPILER in row and row[HOST_COMPILER].name == NVCC:
        reason(output, "nvcc is not allowed as host compiler")
        return False

    if HOST_COMPILER in row and DEVICE_COMPILER in row:
        if NVCC in (row[HOST_COMPILER].name, row[DEVICE_COMPILER].name):
            # Rule: c2
            # related to rule c13
            if row[HOST_COMPILER].name not in (GCC, CLANG):
                reason(output, "only gcc and clang are allowed as nvcc host compiler")
                return False
        else:
            # Rule: c3
            if row[HOST_COMPILER].name != row[DEVICE_COMPILER].name:
                reason(output, "host and device compiler name must be the same (except for nvcc)")
                return False

            # Rule: c4
            if row[HOST_COMPILER].version != row[DEVICE_COMPILER].version:
                reason(
                    output,
                    "host and device compiler version must be the same (except for nvcc)",
                )
                return False

    # now idea, how remove nested blocks without hitting the performance
    # pylint: disable=too-many-nested-blocks
    if DEVICE_COMPILER in row and row[DEVICE_COMPILER].name == NVCC:
        if HOST_COMPILER in row and row[HOST_COMPILER].name == GCC:
            # Rule: c5
            # related to rule b10
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
            # related to rule b11
            if row[DEVICE_COMPILER].version >= pkv.parse("11.3") and row[
                DEVICE_COMPILER
            ].version <= pkv.parse("11.5"):
                reason(
                    output,
                    "clang as host compiler is disabled for nvcc 11.3 to 11.5",
                )
                return False

            # Rule: c6
            # related to rule b12
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

        # Rule: c15
        # related to rule b9
        if (
            ALPAKA_ACC_GPU_CUDA_ENABLE in row
            and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != row[DEVICE_COMPILER].version
        ):
            reason(output, "nvcc and CUDA backend needs to have the same version")
            return False

        # Rule: c16
        # related to rule b14
        if ALPAKA_ACC_GPU_HIP_ENABLE in row and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER:
            reason(output, "nvcc does not support the HIP backend.")
            return False

        # Rule: c17
        # related to rule b15
        if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version != OFF_VER:
            reason(output, "nvcc does not support the SYCL backend.")
            return False

    # Rule: c8
    # related to rule b11
    # clang-cuda 13 and older is not supported
    # this rule will be never used, because of an implementation detail of the covertable library
    # it is not possible to add the clang-cuda versions and filter it out afterwards
    # this rule is only used by bashi-verify
    for compiler in (HOST_COMPILER, DEVICE_COMPILER):
        if (
            compiler in row
            and row[compiler].name == CLANG_CUDA
            and row[compiler].version < pkv.parse("14")
        ):
            reason(output, "all clang versions older than 14 are disabled as CUDA Compiler")
            return False

    for compiler in (HOST_COMPILER, DEVICE_COMPILER):
        if compiler in row and row[compiler].name == HIPCC:
            # Rule: c9
            # related to rule b1
            if (
                ALPAKA_ACC_GPU_HIP_ENABLE in row
                and row[ALPAKA_ACC_GPU_HIP_ENABLE].version == OFF_VER
            ):
                reason(output, "hipcc requires an enabled HIP backend.")
                return False

            # Rule: c10
            # related to rule b2
            if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version != OFF_VER:
                reason(output, "hipcc does not support the SYCL backend.")
                return False

            # Rule: c11
            # related to rule b2
            if (
                ALPAKA_ACC_GPU_CUDA_ENABLE in row
                and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER
            ):
                reason(output, "hipcc does not support the CUDA backend.")
                return False
            # Rule: c19
            # all ROCm images are Ubuntu 20.04 based or newer
            # related to rule d3
            if UBUNTU in row and row[UBUNTU].version < pkv.parse("20.04"):
                reason(
                    output,
                    "ROCm and also the hipcc compiler is not available on Ubuntu older than 20.04",
                )
                return False

        if compiler in row and row[compiler].name == ICPX:
            # Rule: c12
            # related to rule b4
            if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version == OFF_VER:
                reason(output, "icpx requires an enabled SYCL backend.")
                return False

            # Rule: c13
            # related to rule b5
            if (
                ALPAKA_ACC_GPU_HIP_ENABLE in row
                and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER
            ):
                reason(output, "icpx does not support the HIP backend.")
                return False

            # Rule: c14
            # related to rule b6
            if (
                ALPAKA_ACC_GPU_CUDA_ENABLE in row
                and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER
            ):
                reason(output, "icpx does not support the CUDA backend.")
                return False

        if compiler in row and row[compiler].name == CLANG_CUDA:
            # Rule: c15
            # related to rule b16
            if (
                ALPAKA_ACC_GPU_CUDA_ENABLE in row
                and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version == OFF_VER
            ):
                reason(output, "clang-cuda requires an enabled CUDA backend.")
                return False

            if (
                ALPAKA_ACC_GPU_CUDA_ENABLE in row
                and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER
            ):
                # Rule: c16
                # related to rule b17
                # if a clang-cuda version is newer than the latest known clang-cuda version,
                # we needs to assume that it supports every CUDA SDK version
                # pylint: disable=duplicate-code
                if row[compiler].version <= CLANG_CUDA_MAX_CUDA_VERSION[0].clang_cuda:
                    # check if know clang-cuda version supports CUDA SDK version
                    for version_combination in CLANG_CUDA_MAX_CUDA_VERSION:
                        if row[compiler].version >= version_combination.clang_cuda:
                            if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version > version_combination.cuda:
                                reason(
                                    output,
                                    f"clang-cuda {row[compiler].version} does not support "
                                    f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version}.",
                                )
                                return False
                            break

            # Rule: c17
            # related to rule b14
            if (
                ALPAKA_ACC_GPU_HIP_ENABLE in row
                and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER
            ):
                reason(output, "clang-cuda does not support the HIP backend.")
                return False

            # Rule: c18
            # related to rule b15
            if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version != OFF_VER:
                reason(output, "clang-cuda does not support the SYCL backend.")
                return False

        if CXX_STANDARD in row:
            for compiler in (HOST_COMPILER, DEVICE_COMPILER):
                # Rule: c21
                if compiler in row and row[compiler].name == GCC:
                    for gcc_cxx_ver in GCC_CXX_SUPPORT_VERSION:
                        if row[compiler].version >= gcc_cxx_ver.gcc:
                            if row[CXX_STANDARD].version > gcc_cxx_ver.cxx:
                                reason(
                                    output,
                                    f"{compiler} GCC {row[compiler].version} does not support "
                                    f"C++{row[CXX_STANDARD].version}",
                                )
                                return False
                            # break loop otherwise the C++ support of an older GCC version is
                            # applied
                            break
                    # handle case, if GCC version is older, than the oldest defined GCC C++ support
                    # entry
                    if (
                        row[compiler].version < GCC_CXX_SUPPORT_VERSION[-1].gcc
                        and row[CXX_STANDARD].version >= GCC_CXX_SUPPORT_VERSION[-1].cxx
                    ):
                        reason(
                            output,
                            f"{compiler} GCC {row[compiler].version} does not support "
                            f"C++{row[CXX_STANDARD].version}",
                        )
                        return False
    return True
