"""Filter rules handling software dependencies and compiler settings.

All rules implemented in this filter have an identifier that begins with "sw" and follows a number.
Examples: sw1, sw42, sw678 ...

These identifiers are used in the test names, for example, to make it clear which test is testing
which rule.
"""

from typing import Dict, Optional, IO, Callable
import packaging.version as pkv
from typeguard import typechecked
from bashi.types import ParameterValueTuple
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter import FilterBase
from bashi.versions import (
    UBUNTU_HIP_VERSION_RANGE,
    UBUNTU_CUDA_VERSION_RANGE,
    UBUNTU_CLANG_CUDA_SDK_SUPPORT,
)


def _ubuntu_version_to_string(version: pkv.Version) -> str:
    """Returns the Ubuntu version representation correctly. Ubuntu versions
    use a leading 0 in their version scheme for months before October. pkv.parse()`
    parses e.g. the 04 from 20.04 to 4. Therefore the string representation of
    str(pkv.parse(“20.04”)) is `20.4`. This function returns the correct version scheme.
    For Ubuntu `20.04` it is `20.04`.

    Args:
        version (pkv.Version): Ubuntu version

    Returns:
        str: string representation of the Ubuntu version
    """
    return f"{version.major}.{version.minor:02}"


def _pretty_name_compiler(constant: str) -> str:
    """Returns the string representation of the constants HOST_COMPILER and DEVICE_COMPILER in a
    human-readable version.

    Args:
        constant (str): Ether HOST_COMPILER or DEVICE_COMPILER

    Returns:
        str: human-readable string representation of HOST_COMPILER or DEVICE_COMPILER
    """
    if constant == HOST_COMPILER:
        return "host compiler"
    if constant == DEVICE_COMPILER:
        return "device compiler"
    return "unknown compiler type"


class SoftwareDependencyFilter(FilterBase):
    """Filter rules handling software dependencies and compiler settings."""

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
        """Check if given parameter-value-tuple is valid.

        Args:
            row (ParameterValueTuple): parameter-value-tuple to verify.

        Returns:
            bool: True, if parameter-value-tuple is valid.
        """
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-return-statements
        # pylint: disable=too-many-statements

        # Rule: d1
        # GCC 6 and older is not available in Ubuntu 20.04 and newer

        if UBUNTU in row and row[UBUNTU].version >= pkv.parse("20.04"):
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                if compiler_type in row and row[compiler_type].name == GCC:
                    if row[compiler_type].version <= pkv.parse("6"):
                        self.reason(
                            f"{_pretty_name_compiler(compiler_type)} GCC "
                            f"{row[compiler_type].version} is not available in Ubuntu "
                            f"{_ubuntu_version_to_string(row[UBUNTU].version)}",
                        )
                        return False

        # Rule: d2
        # CMAKE 3.19 and older is not available with clang-cuda as device and host compiler

        if CMAKE in row and row[CMAKE].version <= pkv.parse("3.18"):
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                if compiler_type in row and row[compiler_type].name == CLANG_CUDA:
                    self.reason(
                        f"{_pretty_name_compiler(compiler_type)} CLANG_CUDA "
                        "is not available in CMAKE "
                        f"{row[CMAKE].version}",
                    )
                    return False

        if UBUNTU in row:
            # Rule: d3
            # check if a hipcc version is available on an ubuntu version
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                if compiler_type in row and row[compiler_type].name == HIPCC:
                    for ubuntu_hip_range in UBUNTU_HIP_VERSION_RANGE:
                        if (
                            row[compiler_type].version in ubuntu_hip_range.sdk_range
                            and row[UBUNTU].version != ubuntu_hip_range.ubuntu
                        ):
                            self.reason(
                                f"The hipcc {row[compiler_type].version} compiler is not available "
                                f"on the Ubuntu {_ubuntu_version_to_string(row[UBUNTU].version)} "
                                "image.",
                            )
                            return False

            # Rule: d5
            if (
                ALPAKA_ACC_GPU_HIP_ENABLE in row
                and row[ALPAKA_ACC_GPU_HIP_ENABLE].version == ON_VER
            ):
                if RT_AVAILABLE_HIP_SDK_UBUNTU_VER in self.runtime_infos and not self.runtime_infos[
                    RT_AVAILABLE_HIP_SDK_UBUNTU_VER
                ](row[UBUNTU].version):
                    self.reason(
                        f"There is no HIP SDK in input parameter-value-matrix which can be "
                        f"installed on Ubuntu {_ubuntu_version_to_string(row[UBUNTU].version)}"
                    )
                    return False

            # Rule: d6
            if DEVICE_COMPILER in row and row[DEVICE_COMPILER].name == NVCC:
                for ubuntu_cuda_range in UBUNTU_CUDA_VERSION_RANGE:
                    if (
                        row[DEVICE_COMPILER].version in ubuntu_cuda_range.sdk_range
                        and row[UBUNTU].version != ubuntu_cuda_range.ubuntu
                    ):
                        self.reason(
                            f"The nvcc {row[DEVICE_COMPILER].version} compiler is not available "
                            f"on the Ubuntu {_ubuntu_version_to_string(row[UBUNTU].version)} "
                            "image.",
                        )
                        return False

            # Rule: d7
            if (
                ALPAKA_ACC_GPU_CUDA_ENABLE in row
                and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER
            ):
                for ubuntu_cuda_range in UBUNTU_CUDA_VERSION_RANGE:
                    if (
                        row[ALPAKA_ACC_GPU_CUDA_ENABLE].version in ubuntu_cuda_range.sdk_range
                        and row[UBUNTU].version != ubuntu_cuda_range.ubuntu
                    ):
                        self.reason(
                            f"The CUDA SDK {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} is not "
                            "available on the Ubuntu "
                            f"{_ubuntu_version_to_string(row[UBUNTU].version)} image.",
                        )
                        return False

            # Rule: d8
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                if compiler_type in row and row[compiler_type].name == CLANG_CUDA:
                    if row[UBUNTU].version not in UBUNTU_CLANG_CUDA_SDK_SUPPORT:
                        self.reason(
                            "There is no installable CUDA SDK available for "
                            f"Ubuntu {_ubuntu_version_to_string(row[UBUNTU].version)}"
                        )
                        return False

                    if (
                        row[compiler_type].version
                        not in UBUNTU_CLANG_CUDA_SDK_SUPPORT[row[UBUNTU].version]
                    ):
                        self.reason(
                            "There is no compatible CUDA SDK for Clang-CUDA "
                            f"{row[compiler_type].version}, which can be installed on "
                            f"Ubuntu {_ubuntu_version_to_string(row[UBUNTU].version)}"
                        )
                        return False

        return True


@typechecked
def software_dependency_filter_typechecked(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Type-checked version of SoftwareDependencyFilter()(). Type checking has a big performance
    cost, which is why the non type-checked version is used for the pairwise generator.
    """
    return SoftwareDependencyFilter(output=output)(row)
