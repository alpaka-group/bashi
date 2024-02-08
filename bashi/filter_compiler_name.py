"""Filter rules basing on host and device compiler names.

All rules implemented in this filter have an identifier that begins with "n" and follows a number. 
Examples: n1, n42, n678 ...

These identifiers are used in the test names, for example, to make it clear which test is testing 
which rule.
"""

from typing import Optional, IO
from typeguard import typechecked
from bashi.types import Parameter, ParameterValueTuple
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.utils import reason


def get_required_parameters() -> List[Parameter]:
    """Return list of parameters which will be checked in the filter.

    Returns:
        List[Parameter]: list of checked parameters
    """
    return [HOST_COMPILER]


@typechecked
def compiler_name_filter_typechecked(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Type-checked version of compiler_name_filter(). Type checking has a big performance cost,
    which is why the non type-checked version is used for the pairwise generator.
    """
    return compiler_name_filter(row, output)


def compiler_name_filter(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Filter rules basing on host and device compiler names.

    Args:
        row (ParameterValueTuple): parameter-value-tuple to verify.
        output (Optional[IO[str]], optional): Writes the reason in the io object why the parameter
            value tuple does not pass the filter. If None, no information is provided. The default
            value is None.

    Returns:
        bool: True, if parameter-value-tuple is valid.
    """
    # Rule: n1
    # NVCC as HOST_COMPILER is not allow
    # this rule will be never used, because of an implementation detail of the covertable library
    # it is not possible to add NVCC as HOST_COMPILER and filter out afterwards
    # this rule is only used by bashi-verify
    if HOST_COMPILER in row and row[HOST_COMPILER].name == NVCC:
        reason(output, "nvcc is not allowed as host compiler")
        return False

    return True
