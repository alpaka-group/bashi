"""Contains default filter chain and avoids circular import"""

from typing import Callable, Dict
from typeguard import typechecked
from bashi.types import ParameterValueTuple

from bashi.filter import FilterBase
from bashi.filter_compiler import CompilerFilter
from bashi.filter_backend import BackendFilter
from bashi.filter_software_dependency import SoftwareDependencyFilter


# pylint: disable=too-few-public-methods
class FilterChain:
    """Concatenate the bashi filter functors in the default order provide a single functor with a
    single entry point."""

    def __init__(
        self,
        runtime_infos: Dict[str, Callable[..., bool]] | None = None,
        custom_filter: FilterBase = FilterBase(),
    ):
        """Construct new FitlerChain.

        Args:
            runtime_infos (Dict[str, Callable[..., bool]], optional): Runtime infos will be
                constructed depending on the input parameter-value-matrix. The functions are named
                by a string, takes an arbitrary number of arguments and return if the combination of
                the given parameter-values are valid. Defaults to None.
            custom_filter (FilterBase, optional): This functor is added as the last filter level and
                allows the user to add custom filter rules without having to create the entire
                filter chain from scratch. Defaults to FilterBase().
        """
        self.compiler_filter = CompilerFilter(runtime_infos=runtime_infos)
        self.backend_filter = BackendFilter(runtime_infos=runtime_infos)
        self.software_dependency_filter = SoftwareDependencyFilter(runtime_infos=runtime_infos)
        self.custom_filter = custom_filter

    def __call__(self, row: ParameterValueTuple) -> bool:
        return (
            self.compiler_filter(row)
            and self.backend_filter(row)
            and self.software_dependency_filter(row)
            and self.custom_filter(row)
        )


@typechecked
def get_default_filter_chain(
    runtime_infos: Dict[str, Callable[..., bool]] | None = None,
    custom_filter: FilterBase = FilterBase(),
) -> FilterChain:
    """Concatenate the bashi filter functions in the default order and return them as one function
    with a single entry point.

    Args:
        runtime_infos (Dict[str, Callable[..., bool]], optional): Runtime infos will be
                constructed depending on the input parameter-value-matrix. The functions are named
                by a string, takes an arbitrary number of arguments and return if the combination of
                the given parameter-values are valid. Defaults to None.
        custom_filter_function (FilterFunction): This functor is added as the last filter level and
                allows the user to add custom filter rules without having to create the entire
                filter chain from scratch. Defaults to FilterBase().

    Returns:
        FilterFunction: The filter function chain, which can be directly used in bashi.FilterAdapter
    """

    return FilterChain(runtime_infos=runtime_infos, custom_filter=custom_filter)
