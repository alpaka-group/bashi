"""Contains relationships with Clang-CUDA and other parameter-values."""

from typing import List
import packaging.version
from bashi.version.dependencies.base_version_support import VersionSupportBase


# pylint: disable=too-few-public-methods
class ClangCudaSDKSupport(VersionSupportBase):
    """Contains a nvcc version and host compiler version. Does automatically parse the input strings
    to package.version.Version.

    Provides comparision operators for sorting.
    """

    def __init__(self, clang_cuda_version: str, cuda_version: str):
        VersionSupportBase.__init__(self, clang_cuda_version, cuda_version)
        self.clang_cuda: packaging.version.Version = self.version1
        self.cuda: packaging.version.Version = self.version2

    def __str__(self) -> str:
        return f"Clang-CUDA {str(self.clang_cuda)} + CUDA SDK {self.cuda}"


CLANG_CUDA_MAX_CUDA_VERSION: List[ClangCudaSDKSupport] = [
    ClangCudaSDKSupport("7", "9.2"),
    ClangCudaSDKSupport("8", "10.0"),
    ClangCudaSDKSupport("10", "10.1"),
    ClangCudaSDKSupport("12", "11.0"),
    ClangCudaSDKSupport("13", "11.2"),
    ClangCudaSDKSupport("14", "11.5"),
    ClangCudaSDKSupport("16", "11.8"),
    ClangCudaSDKSupport("17", "12.1"),
]
