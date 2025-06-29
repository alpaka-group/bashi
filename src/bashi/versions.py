"""Provides all supported software versions"""

import copy
from typing import Dict, List, Union
from collections import OrderedDict
from typeguard import typechecked
import packaging.version as pkv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ValueName, ValueVersion, ParameterValue, ParameterValueMatrix


class VersionSupportBase:
    """Contains a nvcc version and host compiler version. Does automatically parse the input strings
    to package.version.Version.

    Provides comparision operators for sorting.
    """

    def __init__(self, version1: str, version2: str):
        self.version1 = pkv.parse(version1)
        self.version2 = pkv.parse(version2)

    def __lt__(self, other: "VersionSupportBase") -> bool:
        return self.version1 < other.version1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise TypeError(f"does not support other types than {type(self).__name__}")
        return self.version1 == other.version1 and self.version2 == other.version2


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


# pylint: disable=too-few-public-methods
class CompilerCxxSupport(VersionSupportBase):
    """Contains a compiler version and host compiler version. Does automatically parse the input
    strings to package.version.Version.

    Provides comparision operators for sorting.
    """

    def __init__(self, gcc_version: str, cxx_version: str):
        VersionSupportBase.__init__(self, gcc_version, cxx_version)
        self.compiler: packaging.version.Version = self.version1
        self.cxx: packaging.version.Version = self.version2

    def __str__(self) -> str:
        return f"compiler {str(self.compiler)} + CXX {self.cxx}"


VERSIONS: Dict[str, List[Union[str, int, float]]] = {
    GCC: [8, 9, 10, 11, 12, 13],
    CLANG: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    NVCC: [
        11.0,
        11.1,
        11.2,
        11.3,
        11.4,
        11.5,
        11.6,
        11.7,
        11.8,
        12.0,
        12.1,
        12.2,
        12.3,
    ],
    HIPCC: [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 6.0],
    ICPX: ["2023.1.0", "2023.2.0"],
    UBUNTU: [18.04, 20.04],
    CMAKE: [3.18, 3.19, 3.20, 3.21, 3.22, 3.23, 3.24, 3.25, 3.26],
    BOOST: [
        "1.74.0",
        "1.75.0",
        "1.76.0",
        "1.77.0",
        "1.78.0",
        "1.79.0",
        "1.80.0",
        "1.81.0",
        "1.82.0",
    ],
    CXX_STANDARD: [17, 20],
}
# Clang and Clang-CUDA has the same version numbers
VERSIONS[CLANG_CUDA] = copy.copy(VERSIONS[CLANG])

# define the maximum supported gcc version for a specific nvcc version
# the latest supported nvcc version must be added, even if the supported gcc version does not
# increase
# e.g.:
#   NvccHostSupport("12.3", "12"),
#   NvccHostSupport("12.0", "12"),
#   NvccHostSupport("11.4", "11"),
NVCC_GCC_MAX_VERSION: List[NvccHostSupport] = [
    NvccHostSupport("12.3", "12"),
    NvccHostSupport("12.0", "12"),
    NvccHostSupport("11.4", "11"),
    NvccHostSupport("11.1", "10"),
    NvccHostSupport("11.0", "9"),
    NvccHostSupport("10.1", "8"),
    NvccHostSupport("10.0", "7"),
]
NVCC_GCC_MAX_VERSION.sort(reverse=True)

# define the maximum supported clang version for a specific nvcc version
# the latest supported nvcc version must be added, even if the supported clang version does not
# increase
# e.g.:
#   NvccHostSupport("12.3", "16"),
#   NvccHostSupport("12.2", "15"),
#   NvccHostSupport("12.1", "15"),
NVCC_CLANG_MAX_VERSION: List[NvccHostSupport] = [
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
NVCC_CLANG_MAX_VERSION.sort(reverse=True)

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
CLANG_CUDA_MAX_CUDA_VERSION.sort(reverse=True)

# define the maximum supported cxx version for a specific gcc version
GCC_CXX_SUPPORT_VERSION: List[CompilerCxxSupport] = [
    CompilerCxxSupport("8", "17"),
    CompilerCxxSupport("10", "20"),
    CompilerCxxSupport("11", "23"),
]
GCC_CXX_SUPPORT_VERSION.sort(reverse=True)

# define the maximum supported cxx version for a specific clang version
CLANG_CXX_SUPPORT: List[CompilerCxxSupport] = [
    CompilerCxxSupport("9", "17"),
    CompilerCxxSupport("14", "20"),
    CompilerCxxSupport("17", "23"),
]
CLANG_CXX_SUPPORT.sort(reverse=True)

# define the maximum supported cxx version for a specific nvcc version
NVCC_CXX_SUPPORT_VERSION: List[CompilerCxxSupport] = [
    CompilerCxxSupport("10.0", "14"),
    CompilerCxxSupport("11.0", "17"),
    CompilerCxxSupport("12.0", "20"),
]
NVCC_CXX_SUPPORT_VERSION.sort(reverse=True)


# pylint: disable=too-many-branches
def get_parameter_value_matrix() -> ParameterValueMatrix:
    """Generates a parameter-value-matrix from all supported compilers, softwares and compilation
    configuration.

    Returns:
        ParameterValueMatrix: parameter-value-matrix
    """
    param_val_matrix: ParameterValueMatrix = OrderedDict()

    for compiler_type in [HOST_COMPILER, DEVICE_COMPILER]:
        param_val_matrix[compiler_type] = []
        for sw_name, sw_versions in VERSIONS.items():
            if sw_name in COMPILERS:
                for sw_version in sw_versions:
                    param_val_matrix[compiler_type].append(
                        ParameterValue(sw_name, pkv.parse(str(sw_version)))
                    )

    for backend in BACKENDS:
        if backend == ALPAKA_ACC_GPU_CUDA_ENABLE:
            param_val_matrix[backend] = [ParameterValue(backend, OFF_VER)]
            for cuda_version in VERSIONS[NVCC]:
                param_val_matrix[backend].append(
                    ParameterValue(backend, pkv.parse(str(cuda_version)))
                )
        else:
            param_val_matrix[backend] = [
                ParameterValue(backend, OFF_VER),
                ParameterValue(backend, ON_VER),
            ]

    for other, versions in VERSIONS.items():
        if not other in COMPILERS + BACKENDS:
            param_val_matrix[other] = []
            for version in versions:
                param_val_matrix[other].append(ParameterValue(other, pkv.parse(str(version))))

    return param_val_matrix


@typechecked
def is_supported_version(name: ValueName, version: ValueVersion) -> bool:
    """Check if a specific software version is supported by the bashi library.

    Args:
        name (ValueName): Name of the software, e.g. gcc, boost or ubuntu.
        version (ValueVersion): Version of the software.

    Raises:
        ValueError: If the name of the software is not known.

    Returns:
        bool: True if supported otherwise False.
    """
    known_names: List[ValueName] = list(VERSIONS.keys()) + [CLANG_CUDA] + BACKENDS

    if name not in known_names:
        raise ValueError(f"Unknown software name: {name}")

    local_versions = copy.deepcopy(VERSIONS)

    local_versions[ALPAKA_ACC_GPU_CUDA_ENABLE] = [OFF]
    local_versions[ALPAKA_ACC_GPU_CUDA_ENABLE] += VERSIONS[NVCC]

    for backend_name in BACKENDS:
        if backend_name != ALPAKA_ACC_GPU_CUDA_ENABLE:
            local_versions[backend_name] = [OFF, ON]

    for ver in local_versions[name]:
        if pkv.parse(str(ver)) == version:
            return True

    return False


def get_oldest_supporting_clang_version_for_cuda(
    cuda_version: str,
    clang_cuda_max_cuda_version: List[ClangCudaSDKSupport] = copy.deepcopy(
        CLANG_CUDA_MAX_CUDA_VERSION
    ),
) -> packaging.version.Version:
    """Returns the first and oldest Clang-CUDA version which supports a given CUDA version.
    Args:
        cuda_version (str): CUDA SKD version
        clang_cuda_max_cuda_version (List[ClangCudaSDKSupport], optional): List Clang version with
            the maximum supported CUDA SDK version. Defaults to CLANG_CUDA_MAX_CUDA_VERSION.
    Returns:
        packaging.version.Version: Returns the first and oldest Clang version which supports the
        given CUDA SDK version. Returns version 0, if no version supports the CUDA SDK.
    """
    parsed_cuda_ver = pkv.parse(cuda_version)
    # sort the list by the Clang version starting the smallest version
    # luckily we can assume that the CUDA SDK version is also sorted starting with the smallest
    # version, because a newer Clang version will also support all version like before plus new
    # versions
    clang_cuda_max_cuda_version.sort()

    for sdk_support in clang_cuda_max_cuda_version:
        if sdk_support.cuda >= parsed_cuda_ver:
            return sdk_support.clang_cuda

    # return version 0 as not available
    return OFF_VER
