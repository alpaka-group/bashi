"""Filter rules basing on backend names and versions.

All rules implemented in this filter have an identifier that begins with "b" and follows a number. 
Examples: b1, b42, b678 ...

These identifiers are used in the test names, for example, to make it clear which test is testing 
which rule.
"""

from typing import Optional, IO, List
from typeguard import typechecked
from bashi.types import Parameter, ParameterValueTuple


def get_required_parameters() -> List[Parameter]:
    """Return list of parameters which will be checked in the filter.

    Returns:
        List[Parameter]: list of checked parameters
    """
    return []


@typechecked
def backend_filter_typechecked(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Type-checked version of backend_filter(). Type checking has a big performance cost, which is
    why the non type-checked version is used for the pairwise generator.
    """
    return backend_filter(row, output)


def backend_filter(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Filter rules basing on backend names and versions.

    Args:
        row (ParameterValueTuple): parameter-value-tuple to verify.
        output (Optional[IO[str]], optional): Writes the reason in the io object why the parameter
            value tuple does not pass the filter. If None, no information is provided. The default
            value is None.

    Returns:
        bool: True, if parameter-value-tuple is valid.
    """
    return True
