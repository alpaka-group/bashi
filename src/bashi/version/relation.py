"""Provides relations between different parameter-values and its versions."""

from typing import List, Dict
import packaging.version
import packaging.specifiers
from operator import attrgetter
from bashi.version.dependencies.nvcc import (
    NvccHostSupport,
    NVCC_GCC_MAX_VERSION,
    NVCC_CLANG_MAX_VERSION,
    NVCC_CXX_SUPPORT_VERSION,
)
from bashi.version.dependencies.base_version_support import CompilerCxxSupport, ClangBase
from bashi.version.dependencies.gcc import GCC_CXX_SUPPORT_VERSION
from bashi.version.dependencies.clang import CLANG_CXX_SUPPORT_VERSION
from bashi.version.dependencies.clang_cuda import CLANG_CUDA_MAX_CUDA_VERSION, ClangCudaSDKSupport
from bashi.version.dependencies.icpx import ICPX_CLANG_VERSION
from bashi.version.dependencies.hipcc import HIPCC_CLANG_VERSION
from bashi.version.dependencies.ubuntu import (
    HIP_MIN_UBUNTU,
    CUDA_MIN_UBUNTU,
    SDKUbuntuSupport,
    UbuntuSDKMinMax,
)


# pylint: disable=too-few-public-methods
class VersionRelation:
    """Provides relationships between different parameter values and their versions. The object also
    calculates new relationships based on the input. Enables the extension and modification of
    relationships from outside the Bashi library.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        gcc_cxx_support_version: List[CompilerCxxSupport] | None = None,
        clang_cxx_support_version: List[CompilerCxxSupport] | None = None,
        nvcc_gcc_max_version: List[NvccHostSupport] | None = None,
        nvcc_clang_max_version: List[NvccHostSupport] | None = None,
        nvcc_cxx_support_version: List[CompilerCxxSupport] | None = None,
        clang_cuda_max_cuda_version: List[ClangCudaSDKSupport] | None = None,
        icpx_clang_version: List[ClangBase] | None = None,
        hipcc_clang_version: List[ClangBase] | None = None,
        hip_min_ubuntu: List[SDKUbuntuSupport] | None = None,
        cuda_min_ubuntu: List[SDKUbuntuSupport] | None = None,
    ) -> None:
        self._gcc_cxx_support_version = (
            GCC_CXX_SUPPORT_VERSION if gcc_cxx_support_version is None else gcc_cxx_support_version
        )
        self._gcc_cxx_support_version.sort(reverse=True)

        self._clang_cxx_support_version = (
            CLANG_CXX_SUPPORT_VERSION
            if clang_cxx_support_version is None
            else clang_cxx_support_version
        )
        self._clang_cxx_support_version.sort(reverse=True)

        self._nvcc_gcc_max_version = (
            NVCC_GCC_MAX_VERSION if nvcc_gcc_max_version is None else nvcc_gcc_max_version
        )
        self._nvcc_gcc_max_version.sort(reverse=True)

        self._nvcc_clang_max_version = (
            NVCC_CLANG_MAX_VERSION if nvcc_clang_max_version is None else nvcc_clang_max_version
        )
        self._nvcc_clang_max_version.sort(reverse=True)

        self._nvcc_cxx_support_version = (
            NVCC_CXX_SUPPORT_VERSION
            if nvcc_cxx_support_version is None
            else nvcc_cxx_support_version
        )
        self._nvcc_cxx_support_version.sort(reverse=True)

        self._clang_cuda_max_cuda_version = (
            CLANG_CUDA_MAX_CUDA_VERSION
            if clang_cuda_max_cuda_version is None
            else clang_cuda_max_cuda_version
        )
        self._clang_cuda_max_cuda_version.sort(reverse=True)

        self._max_cuda_sdk_cxx_support = self._get_clang_cuda_cuda_sdk_cxx_support(
            self.get_clang_cuda_cxx_support_version(), self.get_clang_cuda_max_cuda_version()
        )

        self._icpx_cxx_support_version = self._get_clang_base_compiler_cxx_support(
            ICPX_CLANG_VERSION if icpx_clang_version is None else icpx_clang_version,
            self.get_clang_cxx_support_version(),
        )

        self._hipcc_cxx_support_version = self._get_clang_base_compiler_cxx_support(
            HIPCC_CLANG_VERSION if hipcc_clang_version is None else hipcc_clang_version,
            self.get_clang_cxx_support_version(),
        )

        self._ubuntu_hip_version_range = self._get_ubuntu_sdk_min_max(
            HIP_MIN_UBUNTU if hip_min_ubuntu is None else hip_min_ubuntu,
        )

        self._ubuntu_cuda_version_range = self._get_ubuntu_sdk_min_max(
            CUDA_MIN_UBUNTU if cuda_min_ubuntu is None else cuda_min_ubuntu,
        )

        self._ubuntu_clang_cuda_sdk_support = self._get_ubuntu_clang_cuda_sdk_support(
            self.get_ubuntu_cuda_version_range(), self.get_clang_cuda_max_cuda_version()
        )

    def _get_clang_cuda_cuda_sdk_cxx_support(
        self,
        clang_cuda_cxx_support: List[CompilerCxxSupport],
        clang_cuda_max_cuda_support: List[ClangCudaSDKSupport],
    ) -> List[CompilerCxxSupport]:
        """Generate a list containing the latest Clang CUDA version that supports a specific C++
        standard. Two factors must be taken into account. Which C++ standard does a Clang-CUDA version
        support and up to which CUDA SDK version does a Clang-CUDA version support.

        Args:
            clang_cuda_cxx_support (List[CompilerCxxSupport]): Contains which Clang-CUDA version
            supports which C++ standard.
            clang_cuda_max_cuda_support (List[ClangCudaSDKSupport]): Contains which CUDA SDK version
            does a Clang-CUDA version support.

        Returns:
            List[CompilerCxxSupport]: List of Clang-CUDA version with C++ standard. Up to the given
            Clang-CUDA version the related C++ standard is possible. If Clang-CUDA version is younger
            than the latest entry, no C++ standard is possible. If a Clang-CUDA version is older, than
            the oldest version, we assume the older version can have the same C++ standard like the
            oldest defined version. The List is ordered from latest to oldest Clang-CUDA release.
        """
        clang_cuda_cxx_support_sorted: List[CompilerCxxSupport] = sorted(
            clang_cuda_cxx_support, reverse=True
        )
        clang_cuda_max_cuda_support_sorted: List[ClangCudaSDKSupport] = sorted(
            clang_cuda_max_cuda_support, reverse=True
        )
        comb: List[CompilerCxxSupport] = []

        for clang_cuda_cxx in clang_cuda_cxx_support_sorted:
            for clang_cuda_sdk in clang_cuda_max_cuda_support_sorted:
                if clang_cuda_cxx.compiler >= clang_cuda_sdk.clang_cuda:
                    comb.append(
                        CompilerCxxSupport(str(clang_cuda_sdk.cuda), str(clang_cuda_cxx.cxx))
                    )
                    break

        return comb

    def _get_clang_base_compiler_cxx_support(
        self, compiler_clang_mapping: List[ClangBase], clang_cxx_support: List[CompilerCxxSupport]
    ) -> List[CompilerCxxSupport]:
        """Takes a list of compilers based on specific Clang versions and calculates their C++
        support based on the C++ support of the underlying Clang versions.

        Args:
            compiler_clang_mapping (List[ClangBase]): List of Clang-based compiler
            clang_cxx_support (List[CompilerCxxSupport]): List of Clang C++ standard support

        Returns:
            List[CompilerCxxSupport]: List of Clang-based compiler C++ standard support
        """

        compiler_clang_mapping_sorted = sorted(compiler_clang_mapping, reverse=True)
        clang_cxx_support_sorted = sorted(clang_cxx_support, reverse=True)

        compiler_cxx_support: List[CompilerCxxSupport] = []

        for compiler_clang in compiler_clang_mapping_sorted:
            if compiler_clang.clang < clang_cxx_support_sorted[-1].compiler:
                compiler_cxx_support.append(
                    CompilerCxxSupport(
                        str(compiler_clang.compiler), str(clang_cxx_support_sorted[-1].cxx)
                    )
                )
                break
            for clang_cxx in clang_cxx_support_sorted:
                if compiler_clang.clang >= clang_cxx.compiler:
                    compiler_cxx_support.append(
                        CompilerCxxSupport(str(compiler_clang.compiler), str(clang_cxx.cxx))
                    )
                    break

        return compiler_cxx_support

    def _get_ubuntu_sdk_min_max(
        self, sdk_min_ubuntu: List[SDKUbuntuSupport]
    ) -> List[UbuntuSDKMinMax]:
        """Convert an SDKUbuntuSupport object to an UbuntuSDKMinMax object. The SDKUbuntuSupport
        defines a version range implicit and a UbuntuSDKMinMax object explicit.

        Args:
            sdk_min_ubuntu (List[SDKUbuntuSupport]): Input list

        Returns:
            List[UbuntuSDKMinMax]: Output list
        """
        l: List[UbuntuSDKMinMax] = []
        if not sdk_min_ubuntu:
            return l
        sdk_min_ubuntu_sorted = sorted(sdk_min_ubuntu)

        l.append(
            UbuntuSDKMinMax(
                ubuntu=sdk_min_ubuntu_sorted[0].ubuntu,
                sdk_range=packaging.specifiers.SpecifierSet(f"<{sdk_min_ubuntu_sorted[0].sdk}"),
            )
        )

        for i in range(len(sdk_min_ubuntu_sorted) - 1):
            l.append(
                UbuntuSDKMinMax(
                    ubuntu=sdk_min_ubuntu_sorted[i].ubuntu,
                    sdk_range=packaging.specifiers.SpecifierSet(
                        f">={sdk_min_ubuntu_sorted[i].sdk}, <{sdk_min_ubuntu_sorted[i+1].sdk}"
                    ),
                )
            )

        l.append(
            UbuntuSDKMinMax(
                ubuntu=sdk_min_ubuntu_sorted[-1].ubuntu,
                sdk_range=packaging.specifiers.SpecifierSet(f">={sdk_min_ubuntu_sorted[-1].sdk}"),
            )
        )

        return l

    def _get_ubuntu_clang_cuda_sdk_support(
        self,
        ubuntu_cuda_version_range: List[UbuntuSDKMinMax],  # UBUNTU_CUDA_VERSION_RANGE
        clang_cuda_max_cuda_version: List[ClangCudaSDKSupport],  # CLANG_CUDA_MAX_CUDA_VERSION
    ) -> Dict[packaging.version.Version, packaging.specifiers.SpecifierSet]:
        """Calculates since which Clang-CUDA version, there is an supported CUDA SDK which can be
        installed on a specific Ubuntu version.

        Args:
            ubuntu_cuda_version_range (List[UbuntuSDKMinMax]): List with supported CUDA
                versions for a specific Ubuntu version. Defaults to UBUNTU_CUDA_VERSION_RANGE.
            clang_cuda_max_cuda_version (List[ClangCudaSDKSupport]): List which defines the
                maximum CUDA support for a specific Clang-CUDA version. Defaults to
                CLANG_CUDA_MAX_CUDA_VERSION.

        Returns:
            Dict[Version, SpecifierSet]: Dict which shows Ubuntu supports which Clang-CUDA versions.
        """

        support_dict: Dict[packaging.version.Version, packaging.specifiers.SpecifierSet] = {}

        ubuntu_cuda_version_range_sorted = sorted(
            ubuntu_cuda_version_range, key=attrgetter("ubuntu")
        )

        for ub_cuda in ubuntu_cuda_version_range_sorted:
            for clang_cuda_sdk in sorted(clang_cuda_max_cuda_version):
                if clang_cuda_sdk.cuda in ub_cuda.sdk_range:
                    if ub_cuda.ubuntu not in support_dict:
                        support_dict[ub_cuda.ubuntu] = packaging.specifiers.SpecifierSet(
                            f">={clang_cuda_sdk.clang_cuda}"
                        )
                    break

        return support_dict

    def get_gcc_cxx_support_version(self) -> List[CompilerCxxSupport]:
        """Return what is the maximum supported C++ standard for various GCC versions."""
        return self._gcc_cxx_support_version

    def get_clang_cxx_support_version(self) -> List[CompilerCxxSupport]:
        """Return what is the maximum supported C++ standard for various Clang versions."""
        return self._clang_cxx_support_version

    def get_nvcc_gcc_max_version(self) -> List[NvccHostSupport]:
        """Return what is the maximum supported GCC versions for various NVCC versions."""
        return self._nvcc_gcc_max_version

    def get_nvcc_clang_max_version(self) -> List[NvccHostSupport]:
        """Return what is the maximum supported Clang versions for various NVCC versions."""
        return self._nvcc_clang_max_version

    def get_nvcc_cxx_support_version(self) -> List[CompilerCxxSupport]:
        """Return what is the maximum supported C++ standard for various NVCC versions."""
        return self._nvcc_cxx_support_version

    def get_clang_cuda_max_cuda_version(self) -> List[ClangCudaSDKSupport]:
        """Return what is the maximum supported CUDA SDK for various Clang-CUDA versions."""
        return self._clang_cuda_max_cuda_version

    def get_clang_cuda_cxx_support_version(self) -> List[CompilerCxxSupport]:
        """Return what is the maximum supported C++ standard for various Clang-CUDA versions."""
        return self._clang_cxx_support_version

    def get_max_cuda_sdk_cxx_support(self) -> List[CompilerCxxSupport]:
        """
        specify maximum possible C++ standard up to a specific CUDA version
        e.g.
        expected_list: List[CompilerCxxSupport] = [
                   CompilerCxxSupport("12.1", "23"),
                   CompilerCxxSupport("11.5", "20"),
                   CompilerCxxSupport("10.0", "17"),
               ]
        up to CUDA 12.1 C++23 is possible, up to CUDA 11.5 C++20 is possible and up to CUDA 10.0
        C++17 is possible
        """
        return self._max_cuda_sdk_cxx_support

    def get_icpx_cxx_support_version(self) -> List[CompilerCxxSupport]:
        """Return what is the maximum supported C++ standard for various icpx versions."""
        return self._icpx_cxx_support_version

    def get_hipcc_cxx_support_version(self) -> List[CompilerCxxSupport]:
        """Return what is the maximum supported C++ standard for various hipcc versions."""
        return self._hipcc_cxx_support_version

    def get_ubuntu_hip_version_range(self) -> List[UbuntuSDKMinMax]:
        """list of ubuntu version with supported HIP SDKs."""
        return self._ubuntu_hip_version_range

    def get_ubuntu_cuda_version_range(self) -> List[UbuntuSDKMinMax]:
        """list of ubuntu version with supported CUDA SDKs."""
        return self._ubuntu_cuda_version_range

    def get_ubuntu_clang_cuda_sdk_support(
        self,
    ) -> Dict[packaging.version.Version, packaging.specifiers.SpecifierSet]:
        """Since which Clang-CUDA version, there is an supported CUDA SDK which can be installed on
        a specific Ubuntu version.

        Returns:
            Dict[Version, SpecifierSet]: Dict which shows Ubuntu supports which Clang-CUDA versions.
        """
        return self._ubuntu_clang_cuda_sdk_support
