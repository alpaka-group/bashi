"""Contains relationships with Nvcc and other parameter-values."""

from typing import List
import packaging.version
from bashi.version.dependencies.base_version_support import VersionSupportBase


# pylint suggest to use a dataclass, but it does not work because of the cast
# pylint: disable=too-few-public-methods
class NvccHostSupport(VersionSupportBase):
    """Contains a nvcc version and host compiler version. Does automatically parse the input strings
    to package.version.Version.

    Provides comparision operators for sorting.
    """

    def __init__(self, nvcc_version: str, host_version: str):
        VersionSupportBase.__init__(self, nvcc_version, host_version)
        self.nvcc: packaging.version.Version = self.version1
        self.host: packaging.version.Version = self.version2

    def __str__(self) -> str:
        return f"nvcc {str(self.nvcc)} + host version {self.host}"


# define the maximum supported gcc version for a specific nvcc version
# the latest supported nvcc version must be added, even if the supported gcc version does not
# increase
# e.g.:
#   NvccHostSupport("12.3", "12"),
#   NvccHostSupport("12.0", "12"),
#   NvccHostSupport("11.4", "11"),
NVCC_GCC_MAX_VERSION: List[NvccHostSupport] = [
    NvccHostSupport("13.0", "15"),
    NvccHostSupport("12.9", "14"),
    NvccHostSupport("12.6", "13"),
    NvccHostSupport("12.4", "13"),
    NvccHostSupport("12.0", "12"),
    NvccHostSupport("11.4", "11"),
    NvccHostSupport("11.1", "10"),
    NvccHostSupport("11.0", "9"),
    NvccHostSupport("10.1", "8"),
    NvccHostSupport("10.0", "7"),
]

# define the maximum supported clang version for a specific nvcc version
# the latest supported nvcc version must be added, even if the supported clang version does not
# increase
# e.g.:
#   NvccHostSupport("12.3", "16"),
#   NvccHostSupport("12.2", "15"),
#   NvccHostSupport("12.1", "15"),
NVCC_CLANG_MAX_VERSION: List[NvccHostSupport] = [
    NvccHostSupport("13.0", "20"),
    NvccHostSupport("12.9", "19"),
    NvccHostSupport("12.6", "18"),
    NvccHostSupport("12.4", "17"),
    NvccHostSupport("12.3", "16"),
    NvccHostSupport("12.2", "15"),
    NvccHostSupport("12.1", "15"),
    NvccHostSupport("12.0", "14"),
    NvccHostSupport("11.6", "13"),
    NvccHostSupport("11.4", "12"),
    NvccHostSupport("11.2", "11"),
    NvccHostSupport("11.1", "10"),
    NvccHostSupport("11.0", "9"),
    NvccHostSupport("10.1", "8"),
    NvccHostSupport("10.0", "6"),
]
