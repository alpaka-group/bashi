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
from bashi.versions import get_oldest_supporting_clang_version_for_cuda


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
        # Rule: d3
        # all ROCm images are Ubuntu 20.04 based or newer
        # related to rule c19
        if UBUNTU in row and row[UBUNTU].version < pkv.parse("20.04"):
            if (
                ALPAKA_ACC_GPU_HIP_ENABLE in row
                and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER
            ):
                self.reason(
                    "ROCm and also the hipcc compiler "
                    "is not available on Ubuntu "
                    "older than 20.04",
                )
                return False

        # Rule: d4
        # Ubuntu 20.04 and newer is not available with CUDA older than 11

        if UBUNTU in row and row[UBUNTU].version >= pkv.parse("20.04"):
            if (
                ALPAKA_ACC_GPU_CUDA_ENABLE in row
                and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER
            ):
                if row[ALPAKA_ACC_GPU_CUDA_ENABLE].version < pkv.parse("11.0"):
                    self.reason(
                        f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} "
                        "is not available in Ubuntu "
                        f"{_ubuntu_version_to_string(row[UBUNTU].version)}",
                    )
                    return False
            if DEVICE_COMPILER in row and row[DEVICE_COMPILER].name == NVCC:
                if row[DEVICE_COMPILER].version < pkv.parse("11.0"):
                    self.reason(
                        f"NVCC {row[DEVICE_COMPILER].version} "
                        "is not available in Ubuntu "
                        f"{_ubuntu_version_to_string(row[UBUNTU].version)}",
                    )
                    return False
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                if compiler_type in row and row[compiler_type].name == CLANG_CUDA:
                    if row[compiler_type].version < get_oldest_supporting_clang_version_for_cuda(
                        "11.0"
                    ):
                        self.reason(
                            f"{_pretty_name_compiler(compiler_type)}"
                            f" clang-cuda {row[compiler_type].version} "
                            "is not available in Ubuntu "
                            f"{_ubuntu_version_to_string(row[UBUNTU].version)}",
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
