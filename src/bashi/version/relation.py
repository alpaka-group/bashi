"""Provides relations between different parameter-values and its versions."""

from typing import List
from bashi.version.dependencies.clang_cuda import CLANG_CUDA_MAX_CUDA_VERSION, ClangCudaSDKSupport


# pylint: disable=too-few-public-methods
class VersionRelation:
    """Provides relationships between different parameter values and their versions. The object also
    calculates new relationships based on the input. Enables the extension and modification of
    relationships from outside the Bashi library.
    """

    def __init__(
        self, clang_cuda_max_cuda_version: List[ClangCudaSDKSupport] | None = None
    ) -> None:
        self._clang_cuda_max_cuda_version = (
            CLANG_CUDA_MAX_CUDA_VERSION
            if clang_cuda_max_cuda_version is None
            else clang_cuda_max_cuda_version
        )
        self._clang_cuda_max_cuda_version.sort(reverse=True)

    def get_clang_cuda_max_cuda_version(self) -> List[ClangCudaSDKSupport]:
        """Return what is the maximum supported CUDA SDK for a specific Clang-CUDA version."""
        return self._clang_cuda_max_cuda_version
