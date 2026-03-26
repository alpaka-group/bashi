"""Provides relations between different parameter-values and its versions."""

from typing import List
from bashi.version.dependencies.nvcc import (
    NvccHostSupport,
    NVCC_GCC_MAX_VERSION,
    NVCC_CLANG_MAX_VERSION,
)
from bashi.version.dependencies.clang_cuda import CLANG_CUDA_MAX_CUDA_VERSION, ClangCudaSDKSupport


# pylint: disable=too-few-public-methods
class VersionRelation:
    """Provides relationships between different parameter values and their versions. The object also
    calculates new relationships based on the input. Enables the extension and modification of
    relationships from outside the Bashi library.
    """

    def __init__(
        self,
        nvcc_gcc_max_version: List[NvccHostSupport] | None = None,
        nvcc_clang_max_version: List[NvccHostSupport] | None = None,
        clang_cuda_max_cuda_version: List[ClangCudaSDKSupport] | None = None,
    ) -> None:
        self._nvcc_gcc_max_version = (
            NVCC_GCC_MAX_VERSION if nvcc_gcc_max_version is None else nvcc_gcc_max_version
        )
        self._nvcc_gcc_max_version.sort(reverse=True)

        self._nvcc_clang_max_version = (
            NVCC_CLANG_MAX_VERSION if nvcc_clang_max_version is None else nvcc_clang_max_version
        )
        self._nvcc_clang_max_version.sort(reverse=True)

        self._clang_cuda_max_cuda_version = (
            CLANG_CUDA_MAX_CUDA_VERSION
            if clang_cuda_max_cuda_version is None
            else clang_cuda_max_cuda_version
        )
        self._clang_cuda_max_cuda_version.sort(reverse=True)

    def get_nvcc_gcc_max_version(self) -> List[NvccHostSupport]:
        """Return what is the maximum supported GCC versions for various NVCC versions."""
        return self._nvcc_gcc_max_version

    def get_nvcc_clang_max_version(self) -> List[NvccHostSupport]:
        """Return what is the maximum supported Clang versions for various NVCC versions."""
        return self._nvcc_clang_max_version

    def get_clang_cuda_max_cuda_version(self) -> List[ClangCudaSDKSupport]:
        """Return what is the maximum supported CUDA SDK for various Clang-CUDA versions."""
        return self._clang_cuda_max_cuda_version
