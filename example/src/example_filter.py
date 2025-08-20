"""Filter function for the example."""

from typing import Dict, Callable, IO
import packaging.version as pkv
from bashi.types import ParameterValueTuple
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter import FilterBase
from bashi.versions import (
    NvccHostSupport,
    NVCC_GCC_MAX_VERSION,
    NVCC_CLANG_MAX_VERSION,
    NVCC_CXX_SUPPORT_VERSION,
)
from .globals import BUILD_TYPE, CMAKE_RELEASE_VER


class ExampleFilter(FilterBase):
    """Filter function defined by the user. In this case, remove some backend combinations, see
    module documentation."""

    def __init__(
        self,
        runtime_infos: Dict[str, Callable[..., bool]] | None = None,
        output: IO[str] | None = None,
    ):
        super().__init__(runtime_infos, output)

    # pylint: disable=too-many-return-statements
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
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
        gpu_compilers = set([NVCC, CLANG_CUDA, HIPCC, ICPX])
        cpu_compilers = set(COMPILERS) - gpu_compilers

        # the OneAPI CPU and FPGA backend behaves like a GPU backend
        gpu_backends = set(
            [
                ALPAKA_ACC_GPU_CUDA_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
                ALPAKA_ACC_ONEAPI_CPU_ENABLE,
                ALPAKA_ACC_ONEAPI_GPU_ENABLE,
                ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
            ]
        )
        cpu_backends = set(CPU_BACKENDS)

        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            if compiler_type in row and row[compiler_type].name in gpu_compilers:
                for cpu_backend in cpu_backends:
                    if cpu_backend in row and row[cpu_backend].version == ON_VER:
                        self.reason(
                            f"Enabled CPU backend {cpu_backend} and GPU {compiler_type} "
                            f"{row[compiler_type].name} cannot used together."
                        )
                        return False

        if DEVICE_COMPILER in row:
            for cpu_compiler in cpu_compilers:
                if row[DEVICE_COMPILER].name == cpu_compiler:
                    for cpu_backend in cpu_backends:
                        if cpu_backend in row and row[cpu_backend].version == OFF_VER:
                            self.reason(
                                f"The backend {cpu_backend} cannot be disabled if device compiler "
                                f"{cpu_compiler} is used"
                            )
                            return False
                    for gpu_backend in gpu_backends:
                        if gpu_backend in row and row[gpu_backend].version != OFF_VER:
                            self.reason(
                                f"The backend {gpu_backend} cannot be enabled if device compiler "
                                f"{cpu_compiler} is used"
                            )
                            return False

        # pylint: disable=too-many-nested-blocks
        if HOST_COMPILER in row and row[HOST_COMPILER].name in (GCC, CLANG):
            if ALPAKA_ACC_GPU_CUDA_ENABLE in row:
                if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version == OFF_VER:
                    for cpu_backend in cpu_backends:
                        if cpu_backend in row and row[cpu_backend].version == OFF_VER:
                            self.reason(
                                "If the host compiler is {row[HOST_COMPILER].name} and "
                                "the CUDA backend is disable, all CPU backends needs to be enabled."
                                f" {cpu_backend} is disabled."
                            )
                            return False
                if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version == ON_VER:
                    for cpu_backend in cpu_backends:
                        if cpu_backend in row and row[cpu_backend].version == ON_VER:
                            self.reason(
                                "If the host compiler is {row[HOST_COMPILER].name} and "
                                "the CUDA backend is disable, all CPU backends needs to be enabled."
                                f" {cpu_backend} is enabled."
                            )
                            return False

            if (
                CXX_STANDARD in row
                and not ALPAKA_ACC_GPU_CUDA_ENABLE in row
                and not (
                    DEVICE_COMPILER in row
                    or (DEVICE_COMPILER in row and row[DEVICE_COMPILER].name == NVCC)
                )
            ):
                for cpu_backend in cpu_backends:
                    if cpu_backend in row and row[cpu_backend].version == OFF_VER:
                        nvcc_host_compiler_max_version: List[NvccHostSupport] = []
                        if row[HOST_COMPILER].name == GCC:
                            nvcc_host_compiler_max_version = sorted(NVCC_GCC_MAX_VERSION)
                        else:
                            nvcc_host_compiler_max_version = sorted(NVCC_CLANG_MAX_VERSION)
                        max_supported_cuda_version = OFF_VER
                        for nvcc_host_ver in nvcc_host_compiler_max_version:
                            if row[HOST_COMPILER].version <= nvcc_host_ver.host:
                                max_supported_cuda_version = nvcc_host_ver.nvcc
                                break
                        max_support_cxx_ver = OFF_VER
                        for nvcc_cxx_support in NVCC_CXX_SUPPORT_VERSION:
                            if max_supported_cuda_version >= nvcc_cxx_support.compiler:
                                max_support_cxx_ver = nvcc_cxx_support.cxx
                                break
                        if row[CXX_STANDARD].version > max_support_cxx_ver:
                            self.reason(
                                "At least one CPU backend is disabled, therefore host compiler "
                                f"{row[HOST_COMPILER].name}-{row[HOST_COMPILER].version} needs to "
                                "be used together with NVCC. The possible maximum support NVCC "
                                f"version is {max_supported_cuda_version} which supports only up "
                                f"to C++{max_support_cxx_ver}."
                            )
                            return False

        for cpu_backend in cpu_backends:
            if cpu_backend in row and row[cpu_backend].version == ON_VER:
                for cpu_backend2 in cpu_backends:
                    if cpu_backend != cpu_backend2:
                        if cpu_backend2 in row and row[cpu_backend2].version == OFF_VER:
                            self.reason(
                                f"All CPU backends needs to be enabled. {cpu_backend} is "
                                f"enabled, therefore {cpu_backend2} needs to be enabled too."
                            )
                            return False
                for gpu_backend in gpu_backends:
                    if gpu_backend in row and row[gpu_backend].version != OFF_VER:
                        self.reason(
                            "If a single CPU backend is enabled all other gpu backends needs to be "
                            "disabled."
                        )
                        return False

        if UBUNTU in row and RT_AVAILABLE_CUDA_SDK_UBUNTU_VER in self.runtime_infos:
            # Clang and GCC can be used either as CPU backend compiler or as Nvcc host compiler
            # Therefore the combination of enabled or disable CPU backend and host Clang/GCC
            # compiler is valid
            # There is special case, if a Ubuntu version exist, where no CUDA SDK can be installed
            # In this case only the enabled CPU backend is valid for the host compiler Clang/GCC and
            # the specific Ubuntu version
            if HOST_COMPILER in row and row[HOST_COMPILER].name in (GCC, CLANG):
                for cpu_backend in cpu_backends:
                    if (
                        cpu_backend in row
                        and row[cpu_backend].version == OFF_VER
                        and not self.runtime_infos[RT_AVAILABLE_CUDA_SDK_UBUNTU_VER](
                            row[UBUNTU].version
                        )
                    ):
                        return False

        # if nvcc does not support a gcc/clang version, the gcc/clang compiler can be only used as
        # cpu compiler
        for compiler_name, max_supported_version in (
            (GCC, max(comb.host for comb in NVCC_GCC_MAX_VERSION)),
            (CLANG, max(comb.host for comb in NVCC_CLANG_MAX_VERSION)),
        ):
            if (
                HOST_COMPILER in row
                and row[HOST_COMPILER].name == compiler_name
                and row[HOST_COMPILER].version > pkv.parse(str(max_supported_version))
            ):
                for cpu_backend in cpu_backends:
                    if cpu_backend in row and row[cpu_backend].version == OFF_VER:
                        return False
        if (
            CMAKE in row
            and row[CMAKE].version <= pkv.parse("3.25")
            and BUILD_TYPE in row
            and row[BUILD_TYPE].version == CMAKE_RELEASE_VER
        ):
            self.reason(
                "CMake 3.25 does not support CMake Release builds."
                "Only for demonstration. In reality it is working."
            )
            return False

        if self.debug_print != FilterDebugMode.OFF:
            print("passed")
        return True
