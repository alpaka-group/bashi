"""Contains default filter chain and avoids circular import"""

from typing import Callable, Dict
from typeguard import typechecked
import covertable  # type: ignore
import termcolor
from bashi.globals import FilterDebugMode

from bashi.filter import FilterBase
from bashi.filter_compiler import CompilerFilter
from bashi.filter_backend import BackendFilter
from bashi.filter_software_dependency import SoftwareDependencyFilter
from bashi.version.relation import VersionRelation
from bashi.printer import get_str_row_nice
from bashi.row import BashiRow


# pylint: disable=too-few-public-methods
class FilterChain:
    """Concatenate the bashi filter functors in the default order provide a single functor with a
    single entry point."""

    def __init__(
        self,
        version_relation: VersionRelation,
        runtime_infos: Dict[str, Callable[..., bool]] | None = None,
        custom_filter: FilterBase = FilterBase(),
        debug_print: FilterDebugMode = FilterDebugMode.OFF,
    ):
        """Construct new FitlerChain.

        Args:
            version_relation (VersionRelation): Provides information about the relationships between
                the versions of various parameter-values. For example, which GCC version supports
                which C++ standard.
            runtime_infos (Dict[str, Callable[..., bool]], optional): Runtime infos will be
                constructed depending on the input parameter-value-matrix. The functions are named
                by a string, takes an arbitrary number of arguments and return if the combination of
                the given parameter-values are valid. Defaults to None.
            custom_filter (FilterBase, optional): This functor is added as the last filter level and
                allows the user to add custom filter rules without having to create the entire
                filter chain from scratch. Defaults to FilterBase().
            debug_print (FilterDebugMode): Depending on the debug mode, print additional information
                for each row passing the filter function. Defaults to FilterDebugMode.OFF.
        """
        self.compiler_filter = CompilerFilter(
            runtime_infos=runtime_infos, version_relation=version_relation
        )
        self.backend_filter = BackendFilter(
            runtime_infos=runtime_infos, version_relation=version_relation
        )
        self.software_dependency_filter = SoftwareDependencyFilter(
            runtime_infos=runtime_infos, version_relation=version_relation
        )
        self.custom_filter = custom_filter
        self.version = version_relation
        if runtime_infos:
            self.custom_filter.runtime_infos = runtime_infos
        self.debug_print = debug_print

    def __call__(self, row: covertable.main.Row) -> bool:
        bashi_row = BashiRow(row)

        result = (
            self.compiler_filter(bashi_row)
            and self.backend_filter(bashi_row)
            and self.software_dependency_filter(bashi_row)
            and self.custom_filter(bashi_row)
        )

        if self.debug_print != FilterDebugMode.OFF:
            validate_args = self.debug_print == FilterDebugMode.VALIDATOR_ARGS
            row_string = get_str_row_nice(row, bashi_validate=validate_args)
            if result:
                # if the term has no color output or we redirect to a file, we need to add text to
                # show if a row passed
                suffix = "" if termcolor.can_colorize() else "\npassed"
                row_string = termcolor.colored(row_string + suffix, "green")
            else:
                row_string = termcolor.colored(row_string, "red")
            print(row_string)

        return result


@typechecked
def get_default_filter_chain(
    version_relation: VersionRelation,
    debug_print: FilterDebugMode = FilterDebugMode.OFF,
    runtime_infos: Dict[str, Callable[..., bool]] | None = None,
    custom_filter: FilterBase = FilterBase(),
) -> FilterChain:
    """Concatenate the bashi filter functions in the default order and return them as one function
    with a single entry point.

    Args:
        version_relation (VersionRelation): Provides information about the relationships between
                the versions of various parameter-values. For example, which GCC version supports
                which C++ standard.
        debug_print (FilterDebugMode): Depending on the debug mode, print additional information
                for each row passing the filter function. Defaults to FilterDebugMode.OFF.
        runtime_infos (Dict[str, Callable[..., bool]], optional): Runtime infos will be
                constructed depending on the input parameter-value-matrix. The functions are named
                by a string, takes an arbitrary number of arguments and return if the combination of
                the given parameter-values are valid. Defaults to None.
        custom_filter_function (FilterChain): This functor is added as the last filter level and
                allows the user to add custom filter rules without having to create the entire
                filter chain from scratch. Defaults to FilterBase().

    Returns:
        FilterChain: The filter function chain, which can be directly used in bashi.FilterAdapter
    """

    return FilterChain(
        version_relation=version_relation,
        runtime_infos=runtime_infos,
        custom_filter=custom_filter,
        debug_print=debug_print,
    )
