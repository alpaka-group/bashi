"""Filter rules basing on backend names and versions.

All rules implemented in this filter have an identifier that begins with "b" and follows a number.
Examples: b1, b42, b678 ...

These identifiers are used in the test names, for example, to make it clear which test is testing
which rule.
"""

from typing import Dict, Optional, IO, Callable
import packaging.version as pkv
from typeguard import typechecked
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValueTuple
from bashi.versions import NVCC_GCC_MAX_VERSION, NVCC_CLANG_MAX_VERSION, CLANG_CUDA_MAX_CUDA_VERSION
from bashi.filter import FilterBase


# pylint: disable=too-many-branches
# pylint: disable=too-many-return-statements
# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-statements
class BackendFilter(FilterBase):
    """Filter rules basing on backend names and versions."""

    def __init__(
        self,
        runtime_infos: Dict[str, Callable[..., bool]] | None = None,
        output: IO[str] | None = None,
    ):
        super().__init__(runtime_infos, output)

    def __call__(
        self,
        row: ParameterValueTuple,
    ) -> bool:
        """Check if given parameter-value-tuple is valid

        Args:
            row (ParameterValueTuple): parameter-value-tuple to verify.

        Returns:
            bool: True, if parameter-value-tuple is valid.
        """

        if ALPAKA_ACC_GPU_HIP_ENABLE in row and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER:
            # Rule: b1
            # related to rule c9
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                if compiler_type in row and row[compiler_type].name != HIPCC:
                    self.reason("An enabled HIP backend requires hipcc as compiler.")
                    return False

            # Rule: b2
            # related to rule c10
            for one_api_backend in ONE_API_BACKENDS:
                if one_api_backend in row and row[one_api_backend].version != OFF_VER:
                    self.reason("The HIP and SYCL backend cannot be enabled on the same time.")
                    return False

            # Rule: b3
            # related to rule c11
            if (
                ALPAKA_ACC_GPU_CUDA_ENABLE in row
                and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER
            ):
                self.reason("The HIP and CUDA backend cannot be enabled on the same time.")
                return False

        for one_api_backend in ONE_API_BACKENDS:
            if one_api_backend in row and row[one_api_backend].version != OFF_VER:
                # Rule: b4
                # related to rule c12
                for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                    if compiler_type in row and row[compiler_type].name != ICPX:
                        self.reason("An enabled SYCL backend requires icpx as compiler.")
                        return False

                # Rule: b5
                # related to rule c13
                if (
                    ALPAKA_ACC_GPU_HIP_ENABLE in row
                    and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER
                ):
                    self.reason("The SYCL and HIP backend cannot be enabled on the same time.")
                    return False

                # Rule: b6
                # related to rule c14
                if (
                    ALPAKA_ACC_GPU_CUDA_ENABLE in row
                    and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER
                ):
                    self.reason("The SYCL and CUDA backend cannot be enabled on the same time.")
                    return False

                # Rule: b18
                for other_one_api_backends in set(ONE_API_BACKENDS) - set([one_api_backend]):
                    if (
                        other_one_api_backends in row
                        and row[other_one_api_backends].version != OFF_VER
                    ):
                        self.reason("Only one SYCL backend can be enabled.")
                        return False

        if ALPAKA_ACC_GPU_CUDA_ENABLE in row and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version == OFF_VER:
            # Rule: b7
            if DEVICE_COMPILER in row and row[DEVICE_COMPILER].name == NVCC:
                self.reason("CUDA backend needs to be enabled for nvcc")
                return False

            # Rule: b16
            # related to rule c15
            for compiler in (HOST_COMPILER, DEVICE_COMPILER):
                if compiler in row and row[compiler].name == CLANG_CUDA:
                    self.reason(f"CUDA backend needs to be enabled for {compiler} clang-cuda")
                    return False

        if ALPAKA_ACC_GPU_CUDA_ENABLE in row and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER:
            # Rule: b8
            # related to rule c2
            if HOST_COMPILER in row and row[HOST_COMPILER].name in (
                set(COMPILERS) - set([GCC, CLANG, NVCC, CLANG_CUDA])
            ):
                self.reason(
                    f"host-compiler {row[HOST_COMPILER].name} does not support the CUDA backend",
                )
                return False

            # Rule: b9
            # related to rule c15
            if (
                DEVICE_COMPILER in row
                and row[DEVICE_COMPILER].name == NVCC
                and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != row[DEVICE_COMPILER].version
            ):
                self.reason("CUDA backend and nvcc needs to have the same version")
                return False

            if HOST_COMPILER in row and row[HOST_COMPILER].name == GCC:
                # Rule: b10
                # related to rule c5
                # remove all unsupported cuda sdk gcc version combinations
                # define which is the latest supported gcc compiler for a cuda sdk version

                # if a cuda sdk version is not supported by bashi, assume that the version supports
                # the latest gcc compiler version
                if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version <= NVCC_GCC_MAX_VERSION[0].nvcc:
                    # check the maximum supported gcc version for the given nvcc version
                    for nvcc_gcc_comb in NVCC_GCC_MAX_VERSION:
                        if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version >= nvcc_gcc_comb.nvcc:
                            if row[HOST_COMPILER].version > nvcc_gcc_comb.host:
                                self.reason(
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
                    self.reason("clang as host compiler is disabled for CUDA 11.3 to 11.5")
                    return False

                # Rule: b12
                # related to rule c6
                if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version <= NVCC_CLANG_MAX_VERSION[0].nvcc:
                    # check the maximum supported clang version for the given cuda sdk version
                    for nvcc_clang_comb in NVCC_CLANG_MAX_VERSION:
                        if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version >= nvcc_clang_comb.nvcc:
                            if row[HOST_COMPILER].version > nvcc_clang_comb.host:
                                self.reason(
                                    f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} "
                                    f"does not support clang {row[HOST_COMPILER].version}",
                                )
                                return False
                            break

            # Rule: b13
            # related to rule c2
            if DEVICE_COMPILER in row and row[DEVICE_COMPILER].name not in (NVCC, CLANG_CUDA):
                self.reason(f"{row[DEVICE_COMPILER].name} does not support the CUDA backend")
                return False

            # Rule: b14
            # related to rule c30
            if (
                ALPAKA_ACC_GPU_HIP_ENABLE in row
                and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER
            ):
                self.reason("The CUDA and HIP backend cannot be enabled on the same time.")
                return False

            # Rule: b15
            # related to rule c17
            for one_api_backend in ONE_API_BACKENDS:
                if one_api_backend in row and row[one_api_backend].version != OFF_VER:
                    self.reason("The CUDA and SYCL backend cannot be enabled on the same time.")
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
                                if (
                                    row[ALPAKA_ACC_GPU_CUDA_ENABLE].version
                                    > version_combination.cuda
                                ):
                                    self.reason(
                                        f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} is not "
                                        f"supported by Clang-CUDA {row[compiler].version}",
                                    )
                                    return False
                                break

        return True


@typechecked
def backend_filter_typechecked(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Type-checked version of BackendFilter()(). Type checking has a big performance cost, which is
    why the non type-checked version is used for the pairwise generator.
    """
    return BackendFilter(output=output)(row)
