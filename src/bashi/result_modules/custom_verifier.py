"""Filter rules which are not use in bashi. They are designed to be used in user projects."""

from typing import List

from bashi.globals import DEVICE_COMPILER, HOST_COMPILER, ON
from bashi.types import CompilerBackendCombination, ParameterValuePair, ValueName
from bashi.utils import remove_parameter_value_pairs_ranges


def remove_unsupported_compiler_backend_combinations(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    all_used_compilers: List[ValueName],
    all_used_backends: List[ValueName],
    allowed_compiler_backend_combinations: List[CompilerBackendCombination],
):
    """
    For a given list of valid compiler backend combinations, removes all parameter-value-pairs in
    which a compiler and a backend are the parameters. The backend must be enabled and must not be
    part of a CompilerBackendCombination with the corresponding compiler.

    Note: Does not remove disabled backends. This would remove valid pairs.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
        removed_parameter_value_pairs (List[ParameterValuePair]): list with removed
            parameter-value-pairs
        all_used_compilers (List[ValueName]): List of all used compilers.
        all_used_backends (List[ValueName]): List of all used backends.
        allowed_compiler_backend_combinations (List[CompilerBackendCombination]): List of all
            allowed compiler backend combinations.
    """
    supported_backends: dict[ValueName, list[ValueName]] = {name: [] for name in all_used_compilers}
    # the tuple (HOST_COMPILER, DEVICE_COMPILER) has the same ordering like the type CompilerBackendCombination
    # id=0 -> host, id=1 -> device
    for id, compiler_type in enumerate((HOST_COMPILER, DEVICE_COMPILER)):
        for compiler_backend_combination in allowed_compiler_backend_combinations:
            name = ValueName(compiler_backend_combination[id])
            supported_backends[name] += compiler_backend_combination.backends

        for name, backends in supported_backends.items():
            for unsupported_backend in set(all_used_backends) - set(backends):
                remove_parameter_value_pairs_ranges(
                    parameter_value_pairs,
                    removed_parameter_value_pairs,
                    parameter1=compiler_type,
                    value_name1=name,
                    parameter2=unsupported_backend,
                    value_min_version2=ON,
                )
