"""Provide class to build bashi based validator app."""

import io
import sys
from typing import Dict, List
from collections import OrderedDict
import packaging.version
from typeguard import typechecked
from bashi.types import ParameterValue, ParameterValueTuple
from bashi.versions import is_supported_version
from bashi.filter import FilterBase
from bashi.filter_compiler import CompilerFilter
from bashi.filter_backend import BackendFilter
from bashi.filter_software_dependency import SoftwareDependencyFilter
from bashi.exceptions import BashiUnknownVersion
from .arguments import get_validator_args, ArgumentAlias, VersionCheck
from .utils import cs, Color


class Validator:
    """Constructs a parameter-value-tuple from command line arguments and checks if it passes
    the filter stages.
    """

    def __init__(self, args: List[str] | None = None, silent: bool = False):
        """Setup default configuration for bashi filter rules.

        Args:
            args (List[str] | None, optional): Set command line arguments manually. Defaults to
                None.
            silent (bool, optional): If True, disable terminal output. Defaults to False.
        """
        self.parser, self.argument_alias, self.param_order = get_validator_args()
        self.filter_stages: List[FilterBase] = [
            CompilerFilter(),
            BackendFilter(),
            SoftwareDependencyFilter(),
        ]
        self.known_version: Dict[str, List[packaging.version.Version]] = {}
        if args:
            self.args = args
        else:
            self.args = sys.argv[1:]
        self.silent = silent

    @typechecked
    def add_software_version_parameter(self, name: str, help_text: str, short_name: str = ""):
        """Add new software name to application argument list.

        Args:
            name (str): Name of the software
            help_text (str): Description text of software
            short_name (str, optional): Short argument name as alias. Defaults to "".
        """
        self.parser.add_argument(f"--{name}", type=str, action=VersionCheck, help=help_text)

        argument_names: List[str] = [f"--{name}"]

        if short_name != "":
            argument_names.append(f"--{short_name}")

        self.argument_alias[name.replace("-", "_")] = ArgumentAlias(argument_names, name)

    @typechecked
    def add_known_version(self, name: str, versions: List[str]):
        """Add known versions for a software, which was added via `add_software_version_parameter`.
            Disable warning for unsupported software versions.

        Args:
            name (str): Software name
            versions (List[str]): Software versions
        """
        self.known_version[name] = [packaging.version.parse(ver) for ver in versions]

    @typechecked
    def add_custom_filter(self, custom_filter: FilterBase):
        """Add custom software filter stage. Is applied after the default filter stages of bashi.

        Args:
            custom_filter (FilterBase): Custom filter
        """
        self.filter_stages.append(custom_filter)

    @typechecked
    def _print(self, msg: str):
        """Print message only if silent flag is false.

        Args:
            msg (str): Message to be printed.
        """
        if not self.silent:
            print(msg)

    @typechecked
    def _check_single_filter(
        self,
        filter_func: FilterBase,
        row: ParameterValueTuple,
    ) -> bool:
        """Check if row passes a filter function.

        Args:
            filter_func (Callable[[ParameterValueTuple, Optional[IO[str]]], bool]): The filter
                function
            row (ParameterValueTuple): row with parameter-value-tuples
            required_parameter (List[str]): list of parameters, which will be used in the filter
                rule

        Returns:
            bool: True if the row passes the filter.
        """
        # get name of the filter for command line output.
        filter_name: str = filter_func.__class__.__name__

        filter_func.output = io.StringIO()

        if filter_func(row):
            self._print(cs(f"{filter_name}() returns True", Color.GREEN))
            return True

        self._print(cs(f"{filter_name}() returns False", Color.RED))
        if filter_func.output.getvalue() != "":
            self._print("  " + filter_func.output.getvalue())
        return False

    @typechecked
    def _check_filter_chain(self, row: ParameterValueTuple) -> bool:
        """Test a row with the bashi default filter chain.

        Args:
            row (ParameterValueTuple): row to test

        Returns:
            bool: True if row passes all filters
        """
        all_true = 0
        for filter_stage in self.filter_stages:
            all_true += int(
                self._check_single_filter(
                    filter_stage,
                    row,
                )
            )

        return all_true == len(self.filter_stages)

    def validate(self) -> bool:
        """Construct parameter-value-tuple from the application arguments and check if it passes
        the bashi and custom filter.

        Returns:
            bool: Return True if all filter stages are passed
        """
        args = self.parser.parse_args(args=self.args)

        row: ParameterValueTuple = OrderedDict()

        # Add parameter-values in the order in which they are passed via arguments
        for param_arg in self.param_order:
            for arg, alias in self.argument_alias.items():
                if param_arg in alias.alias:
                    if getattr(args, arg) is not None:
                        if arg in ("host_compiler", "device_compiler"):
                            row[alias.parameter] = getattr(args, arg)
                        else:
                            row[alias.parameter] = ParameterValue(
                                alias.parameter, getattr(args, arg)
                            )

        for val_name, val_version in row.values():
            known_software = (
                val_name in self.known_version and val_version in self.known_version[val_name]
            )
            if not known_software:
                try:
                    known_software = is_supported_version(val_name, val_version)
                except BashiUnknownVersion:
                    known_software = False
                except Exception as e:
                    raise e

            if not known_software:
                self._print(
                    cs(
                        f"WARNING: {val_name} {val_version} is not officially supported.",
                        Color.YELLOW,
                    )
                )

        return self._check_filter_chain(row)
