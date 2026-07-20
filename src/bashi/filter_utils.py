"""Provide different filter rules, which can be used in a project but are not mandatory, because
the restriction has no technical reason. For example restrict the number of backend combinations to
reduce the number of generated CI jobs."""

from typing import List

from bashi.row import BashiRow
from bashi.types import ValueName, CompilerBackendCombination
from bashi.globals import DEVICE_COMPILER, HOST_COMPILER, OFF_VER


def all_backends_fine(
    row: BashiRow,
    backends: List[ValueName],
    all_available_backends: List[ValueName],
) -> bool:
    """Check if the combination of backends in a row is corresponding to at least one valid
    combination of backends.

    Args:
        row (BashiRow): row with backends
        backends (List[ValueName]): Backends which needs to be enabled.
        all_available_backends (List[ValueName]): All available backends. If a backend is not in the
            backends list, but in this list, it needs to be disabled.

    Returns:
        bool: True if all enabled backends of the `row` are defined in `backends` and all disabled
            backends are defined in `all_available_backends`.
    """
    for backend in all_available_backends:
        if backend in row:
            if backend in backends:
                if row[backend].version == OFF_VER:
                    return False
            else:
                if row[backend].version != OFF_VER:
                    return False

    return True


def get_valid_compiler_backend_combinations(
    row: BashiRow,
    allowed_backend_combinations: List[CompilerBackendCombination],
    all_available_backends: List[str],
) -> List[CompilerBackendCombination]:
    """Return a list of all possible compiler and backend combinations, which are still possible
    for the given row.

    Args:
        row (BashiRow): parameter-value-tuple
        backends (List[ValueName]): Backends which needs to be enabled.
        all_available_backends (List[ValueName]): All available backends. If a backend is not in the
            backends list, but in this list, it needs to be disabled.

    Returns:
        List[CompilerBackendComb]: List if possible backends combinations
    """
    valid_combs: List[CompilerBackendCombination] = []
    for comb in allowed_backend_combinations:
        host_compiler, device_compiler, backends = comb
        if row[HOST_COMPILER].name != host_compiler:
            continue
        if row[DEVICE_COMPILER].name != device_compiler:
            continue
        if all_backends_fine(row, backends, all_available_backends):
            valid_combs.append(comb)

    return valid_combs
