"""Utility functions for software versions"""

from typing import Dict, List, Union
import copy
from collections import OrderedDict
import packaging
from typeguard import typechecked
from bashi.types import ValueName, ValueVersion, ParameterValue, ParameterValueMatrix
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.version import VERSIONS
from bashi.exceptions import BashiUnknownVersion


# pylint: disable=too-many-branches
def get_parameter_value_matrix(
    software_versions: Dict[str, List[Union[str, int, float]]] | None = None,
    backends: List[str] | None = None,
) -> ParameterValueMatrix:
    """Generates a parameter-value-matrix from all supported compilers, softwares and compilation
    configuration.

    Args:
        software_versions (Dict[str, List[Union[str, int, float]]] | None, optional): Dict of
            software version which will be used to generate the parameter-value-matrix. The default
            value is bashi.globals.VERSION. Defaults to None.
        backends (List[str] | None, optional): List of backend names which will be used to generate
        the parameter-value-matrix. The default value is bashi.globals.BACKENDS Defaults to None.

    Returns:
        ParameterValueMatrix: parameter-value-matrix
    """
    if software_versions is None:
        software_versions = VERSIONS
    if backends is None:
        backends = BACKENDS

    param_val_matrix: ParameterValueMatrix = OrderedDict()

    for compiler_type in [HOST_COMPILER, DEVICE_COMPILER]:
        compilers: List[ParameterValue] = []
        for sw_name, sw_versions in software_versions.items():
            if sw_name in COMPILERS:
                for sw_version in sw_versions:
                    compilers.append(
                        ParameterValue(sw_name, packaging.version.parse(str(sw_version)))
                    )
        if len(compilers) > 0:
            param_val_matrix[compiler_type] = compilers

    for backend in backends:
        if backend == ALPAKA_ACC_GPU_CUDA_ENABLE:
            param_val_matrix[backend] = [ParameterValue(backend, OFF_VER)]
            for cuda_version in software_versions[NVCC]:
                param_val_matrix[backend].append(
                    ParameterValue(backend, packaging.version.parse(str(cuda_version)))
                )
        else:
            param_val_matrix[backend] = [
                ParameterValue(backend, OFF_VER),
                ParameterValue(backend, ON_VER),
            ]

    for other, versions in software_versions.items():
        if not other in COMPILERS + BACKENDS:
            param_val_matrix[other] = []
            for version in versions:
                param_val_matrix[other].append(
                    ParameterValue(other, packaging.version.parse(str(version)))
                )

    return param_val_matrix


@typechecked
def is_supported_version(name: ValueName, version: ValueVersion) -> bool:
    """Check if a specific software version is supported by the bashi library.

    Args:
        name (ValueName): Name of the software, e.g. gcc, boost or ubuntu.
        version (ValueVersion): Version of the software.

    Raises:
        ValueError: If the name of the software is not known.

    Returns:
        bool: True if supported otherwise False.
    """
    known_names: List[ValueName] = list(VERSIONS.keys()) + [CLANG_CUDA] + BACKENDS

    if name not in known_names:
        raise BashiUnknownVersion(f"Unknown software name: {name}")

    local_versions = copy.deepcopy(VERSIONS)

    local_versions[ALPAKA_ACC_GPU_CUDA_ENABLE] = [OFF]
    local_versions[ALPAKA_ACC_GPU_CUDA_ENABLE] += VERSIONS[NVCC]

    for backend_name in BACKENDS:
        if backend_name != ALPAKA_ACC_GPU_CUDA_ENABLE:
            local_versions[backend_name] = [OFF, ON]

    for ver in local_versions[name]:
        parsed_version = packaging.version.parse(str(ver))
        # in case of CMAKE, we don't care about the patch level
        if name == CMAKE:
            parsed_version = packaging.version.parse(
                f"{parsed_version.major}.{parsed_version.minor}"
            )
            version = packaging.version.parse(f"{version.major}.{version.minor}")
        if parsed_version == version:
            return True

    return False
