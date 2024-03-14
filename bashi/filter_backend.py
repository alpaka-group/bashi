"""Filter rules basing on backend names and versions.

All rules implemented in this filter have an identifier that begins with "b" and follows a number. 
Examples: b1, b42, b678 ...

These identifiers are used in the test names, for example, to make it clear which test is testing 
which rule.
"""

from typing import Optional, IO
from typeguard import typechecked
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValueTuple
from bashi.utils import reason


@typechecked
def backend_filter_typechecked(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Type-checked version of backend_filter(). Type checking has a big performance cost, which is
    why the non type-checked version is used for the pairwise generator.
    """
    return backend_filter(row, output)


# pylint: disable=too-many-branches
# pylint: disable=too-many-return-statements
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

    if ALPAKA_ACC_GPU_HIP_ENABLE in row and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER:
        # Rule: b1
        # related to rule c9
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            if compiler_type in row and row[compiler_type].name != HIPCC:
                reason(output, "An enabled HIP backend requires hipcc as compiler.")
                return False

        # Rule: b2
        # related to rule c10
        if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version != OFF_VER:
            reason(output, "The HIP and SYCL backend cannot be enabled on the same time.")
            return False

        # Rule: b3
        # related to rule c11
        if ALPAKA_ACC_GPU_CUDA_ENABLE in row and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER:
            reason(output, "The HIP and CUDA backend cannot be enabled on the same time.")
            return False

    if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version != OFF_VER:
        # Rule: b4
        # related to rule c12
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            if compiler_type in row and row[compiler_type].name != ICPX:
                reason(output, "An enabled SYCL backend requires icpx as compiler.")
                return False

        # Rule: b5
        # related to rule c13
        if ALPAKA_ACC_GPU_HIP_ENABLE in row and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER:
            reason(output, "The SYCL and HIP backend cannot be enabled on the same time.")
            return False

        # Rule: b6
        # related to rule c14
        if ALPAKA_ACC_GPU_CUDA_ENABLE in row and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER:
            reason(output, "The SYCL and CUDA backend cannot be enabled on the same time.")
            return False

    return True
