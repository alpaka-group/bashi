"""Contains default filter chain and avoids circular import"""

from typeguard import typechecked
from bashi.types import FilterFunction

from bashi.filter_compiler_name import compiler_name_filter
from bashi.filter_compiler_version import compiler_version_filter
from bashi.filter_backend import backend_filter
from bashi.filter_software_dependency import software_dependency_filter


@typechecked
def get_default_filter_chain(
    custom_filter_function: FilterFunction = lambda _: True,
) -> FilterFunction:
    """Concatenate the bashi filter functions in the default order and return them as one function
    with a single entry point.

    Args:
        custom_filter_function (FilterFunction): This function is added as the last filter level and
            allows the user to add custom filter rules without having to create the entire filter
            chain from scratch. Defaults to lambda_:True.

    Returns:
        FilterFunction: The filter function chain, which can be directly used in bashi.FilterAdapter
    """
    return (
        lambda row: compiler_name_filter(row)
        and compiler_version_filter(row)
        and backend_filter(row)
        and software_dependency_filter(row)
        and custom_filter_function(row)
    )
