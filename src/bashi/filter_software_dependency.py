"""Filter rules handling software dependencies and compiler settings.

All rules implemented in this filter have an identifier that begins with "sw" and follows a number. 
Examples: sw1, sw42, sw678 ...

These identifiers are used in the test names, for example, to make it clear which test is testing 
which rule.
"""

from typing import Optional, IO
import packaging.version as pkv
from typeguard import typechecked
from bashi.types import ParameterValueTuple
from bashi.globals import DEVICE_COMPILER, HOST_COMPILER, GCC, UBUNTU, CLANG_CUDA, CMAKE
from bashi.utils import reason


def __ubuntu_version_to_string(version: pkv.Version) -> str:
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


def __pretty_name_compiler(constant: str) -> str:
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


@typechecked
def software_dependency_filter_typechecked(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Type-checked version of software_dependency_filter(). Type checking has a big performance
    cost, which is why the non type-checked version is used for the pairwise generator.
    """
    return software_dependency_filter(row, output)


def software_dependency_filter(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Filter rules handling software dependencies and compiler settings.

    Args:
        row (ParameterValueTuple): parameter-value-tuple to verify.
        output (Optional[IO[str]], optional): Writes the reason in the io object why the parameter
            value tuple does not pass the filter. If None, no information is provided. The default
            value is None.

    Returns:
        bool: True, if parameter-value-tuple is valid.
    """

    # Rule: d1
    # GCC 6 and older is not available in Ubuntu 20.04 and newer

    if UBUNTU in row and row[UBUNTU].version >= pkv.parse("20.04"):
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            if compiler_type in row and row[compiler_type].name == GCC:
                if row[compiler_type].version <= pkv.parse("6"):
                    reason(
                        output,
                        f"{__pretty_name_compiler(compiler_type)} GCC {row[compiler_type].version} "
                        "is not available in Ubuntu "
                        f"{__ubuntu_version_to_string(row[UBUNTU].version)}",
                    )
                    return False

    # Rule: d2
    # CMAKE 3.19 and older is not available with clang cuda as device and host compiler

    if CMAKE in row and row[CMAKE].version <= pkv.parse("3.18"):
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            if compiler_type in row and row[compiler_type].name == CLANG_CUDA:
                reason(
                    output,
                    f"{__pretty_name_compiler(compiler_type)} CLANG_CUDA "
                    "is not available in CMAKE "
                    f"{row[CMAKE].version}",
                )
                return False

    return True
