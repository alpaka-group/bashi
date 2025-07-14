"""Base class for filter functors."""

from typing import Dict, Callable, Optional, IO
from bashi.types import ParameterValueTuple


class FilterBase:
    """Base class for a filter functor. A filter functor object behaves like a function. The
    __call__ function implements the required interface of a filter function for the pair-wise
    testing library. The functor allows additional “arguments” to be added to the filter function
    without changing the required function interface, which takes only one parameter-value-tuple.
    """

    def __init__(
        self,
        runtime_infos: Dict[str, Callable[..., bool]] | None = None,
        output: Optional[IO[str]] = None,
    ):
        """Construct new FilterBase object.

        Args:
            runtime_infos (Dict[str, Callable[..., bool]], optional): Runtime infos will be
                constructed depending on the input parameter-value-matrix. The functions are named
                by a string, takes an arbitrary number of arguments and return if the combination of
                the given parameter-values are valid. Defaults to None.
            output (Optional[IO[str]], optional): Write the message to output if it is not None.
                This function is used in filter functions to print additional information about
                filter decisions. Defaults to None.
        """
        self.runtime_infos: Dict[str, Callable[..., bool]] = {}
        if runtime_infos:
            self.runtime_infos = runtime_infos
        self.output = output

    def reason(self, msg: str):
        """Write the message to output if it is not None. This function is used
        in filter functions to print additional information about filter decisions.

        Args:
            msg (str): the message
        """
        if self.output:
            print(
                msg,
                file=self.output,
                end="",
            )

    def __call__(self, _row: ParameterValueTuple) -> bool:
        """Implement the filter rules

        Args:
            _row (ParameterValueTuple): parameter-value-tuple

        Returns:
            bool: Return True if parameter-value-tuple passes the filter, otherwise false
        """
        return True
