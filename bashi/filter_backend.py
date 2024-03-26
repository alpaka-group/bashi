"""Filter rules basing on backend names and versions.

All rules implemented in this filter have an identifier that begins with "b" and follows a number. 
Examples: b1, b42, b678 ...

These identifiers are used in the test names, for example, to make it clear which test is testing 
which rule.
"""

from typing import Optional, IO
import packaging.version as pkv
from typeguard import typechecked
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValueTuple
from bashi.versions import NVCC_GCC_MAX_VERSION, NVCC_CLANG_MAX_VERSION, CLANG_CUDA_MAX_CUDA_VERSION

from bashi.utils import reason


@typechecked
def backend_filter_typechecked(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Type-checked version of backend_filter(). Type checking has a big performance cost, which is
    why the non type-checked version is used for the pairwise generator.
    """
    return backend_filter(row, output)


# pylint: disable=too-many-branches
# pylint: disable=too-many-return-statements
# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-statements
def backend_filter(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Filter rules basing on backend names and versions.

    Args:
        row (ParameterValueTuple): parameter-value-tuple to verify.
        output (Optional[IO[str]], optional): Writes the reason in the io object why the parameter
            value tuple does not pass the filter. If None, no information is provided. The default
            value is None.

    Returns:
        bool: True, if parameter-value-tuple is valid.
    """

    if ALPAKA_ACC_GPU_HIP_ENABLE in row and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER:
        # Rule: b1
        # related to rule c9
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            if compiler_type in row and row[compiler_type].name != HIPCC:
                reason(output, "An enabled HIP backend requires hipcc as compiler.")
                return False

        # Rule: b2
        # related to rule c10
        if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version != OFF_VER:
            reason(output, "The HIP and SYCL backend cannot be enabled on the same time.")
            return False

        # Rule: b3
        # related to rule c11
        if ALPAKA_ACC_GPU_CUDA_ENABLE in row and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER:
            reason(output, "The HIP and CUDA backend cannot be enabled on the same time.")
            return False

    if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version != OFF_VER:
        # Rule: b4
        # related to rule c12
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            if compiler_type in row and row[compiler_type].name != ICPX:
                reason(output, "An enabled SYCL backend requires icpx as compiler.")
                return False

        # Rule: b5
        # related to rule c13
        if ALPAKA_ACC_GPU_HIP_ENABLE in row and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER:
            reason(output, "The SYCL and HIP backend cannot be enabled on the same time.")
            return False

        # Rule: b6
        # related to rule c14
        if ALPAKA_ACC_GPU_CUDA_ENABLE in row and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER:
            reason(output, "The SYCL and CUDA backend cannot be enabled on the same time.")
            return False

    if ALPAKA_ACC_GPU_CUDA_ENABLE in row and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version == OFF_VER:
        # Rule: b7
        if DEVICE_COMPILER in row and row[DEVICE_COMPILER].name == NVCC:
            reason(output, "CUDA backend needs to be enabled for nvcc")
            return False

        # Rule: b16
        # related to rule c15
        for compiler in (HOST_COMPILER, DEVICE_COMPILER):
            if compiler in row and row[compiler].name == CLANG_CUDA:
                reason(output, f"CUDA backend needs to be enabled for {compiler} clang-cuda")
                return False

    if ALPAKA_ACC_GPU_CUDA_ENABLE in row and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER:
        # Rule: b8
        # related to rule c2
        if HOST_COMPILER in row and row[HOST_COMPILER].name in (
            set(COMPILERS) - set([GCC, CLANG, NVCC, CLANG_CUDA])
        ):
            reason(
                output, f"host-compiler {row[HOST_COMPILER].name} does not support the CUDA backend"
            )
            return False

        # Rule: b9
        # related to rule c15
        if (
            DEVICE_COMPILER in row
            and row[DEVICE_COMPILER].name == NVCC
            and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != row[DEVICE_COMPILER].version
        ):
            reason(output, "CUDA backend and nvcc needs to have the same version")
            return False

        if HOST_COMPILER in row and row[HOST_COMPILER].name == GCC:
            # Rule: b10
            # related to rule c5
            # remove all unsupported cuda sdk gcc version combinations
            # define which is the latest supported gcc compiler for a cuda sdk version

            # if a cuda sdk version is not supported by bashi, assume that the version supports the
            # latest gcc compiler version
            if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version <= NVCC_GCC_MAX_VERSION[0].nvcc:
                # check the maximum supported gcc version for the given nvcc version
                for nvcc_gcc_comb in NVCC_GCC_MAX_VERSION:
                    if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version >= nvcc_gcc_comb.nvcc:
                        if row[HOST_COMPILER].version > nvcc_gcc_comb.host:
                            reason(
                                output,
                                f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} "
                                f"does not support gcc {row[HOST_COMPILER].version}",
                            )
                            return False
                        break

        if HOST_COMPILER in row and row[HOST_COMPILER].name == CLANG:
            # Rule: b11
            # related to rule c8
            if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version >= pkv.parse("11.3") and row[
                ALPAKA_ACC_GPU_CUDA_ENABLE
            ].version <= pkv.parse("11.5"):
                reason(
                    output,
                    "clang as host compiler is disabled for CUDA 11.3 to 11.5",
                )
                return False

            # Rule: b12
            # related to rule c6
            if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version <= NVCC_CLANG_MAX_VERSION[0].nvcc:
                # check the maximum supported clang version for the given cuda sdk version
                for nvcc_clang_comb in NVCC_CLANG_MAX_VERSION:
                    if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version >= nvcc_clang_comb.nvcc:
                        if row[HOST_COMPILER].version > nvcc_clang_comb.host:
                            reason(
                                output,
                                f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} "
                                f"does not support clang {row[HOST_COMPILER].version}",
                            )
                            return False
                        break

        # Rule: b13
        # related to rule c2
        if DEVICE_COMPILER in row and row[DEVICE_COMPILER].name not in (NVCC, CLANG_CUDA):
            reason(output, f"{row[DEVICE_COMPILER].name} does not support the CUDA backend")
            return False

        # Rule: b14
        # related to rule c16
        if ALPAKA_ACC_GPU_HIP_ENABLE in row and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER:
            reason(output, "The CUDA and HIP backend cannot be enabled on the same time.")
            return False

        # Rule: b15
        # related to rule c17
        if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version != OFF_VER:
            reason(output, "The CUDA and SYCL backend cannot be enabled on the same time.")
            return False

        # Rule: b17
        # related to rule c16
        for compiler in (HOST_COMPILER, DEVICE_COMPILER):
            if compiler in row and row[compiler].name == CLANG_CUDA:
                # if a clang-cuda version is newer than the latest known clang-cuda version,
                # we needs to assume that it supports every CUDA SDK version
                if row[compiler].version <= CLANG_CUDA_MAX_CUDA_VERSION[0].clang_cuda:
                    for version_combination in CLANG_CUDA_MAX_CUDA_VERSION:
                        if row[compiler].version >= version_combination.clang_cuda:
                            if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version > version_combination.cuda:
                                reason(
                                    output,
                                    f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} is not "
                                    f"supported by Clang-CUDA {row[compiler].version}",
                                )
                                return False
                            break

    return True
