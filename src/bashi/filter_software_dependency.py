"""Filter rules handling software dependencies and compiler settings.

All rules implemented in this filter have an identifier that begins with "sw" and follows a number. 
Examples: sw1, sw42, sw678 ...

These identifiers are used in the test names, for example, to make it clear which test is testing 
which rule.
"""

import packaging.version as pkv
from bashi.globals import *
from typing import Optional, IO
from typeguard import typechecked
from bashi.types import ParameterValueTuple


@typechecked
def software_dependency_filter_typechecked(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Type-checked version of software_dependency_filter(). Type checking has a big performance
    cost, which is why the non type-checked version is used for the pairwise generator.
    """
    return software_dependency_filter(row, output)


# TODO(SimeonEhrig): remove disable=unused-argument
# only required for the CI at the moment
def software_dependency_filter(
    row: ParameterValueTuple,  # pylint: disable=unused-argument
    output: Optional[IO[str]] = None,  # pylint: disable=unused-argument
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

    for compiler in (HOST_COMPILER, DEVICE_COMPILER):
        if compiler in row and row[compiler].name == GCC:
            if row[compiler].version == pkv.parse("6"):
                if UBUNTU in row and row[UBUNTU].version == pkv.parse("20.04"):
                    return False

    return True
