"""Provides all supported software versions"""

import copy
from typing import Dict, List, Union
from collections import OrderedDict
from typeguard import typechecked
import packaging.version as pkv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ValueName, ValueVersion, ParameterValue, ParameterValueMatrix
from bashi.exceptions import BashiUnknownVersion


VERSIONS: Dict[str, List[Union[str, int, float]]] = {
    GCC: [8, 9, 10, 11, 12, 13],
    CLANG: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    NVCC: [
        11.0,
        11.1,
        11.2,
        11.3,
        11.4,
        11.5,
        11.6,
        11.7,
        11.8,
        12.0,
        12.1,
        12.2,
        12.3,
        12.4,
        12.5,
        12.6,
        12.7,
        12.8,
        12.9,
        13.0,
    ],
    HIPCC: [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 6.0, 6.1, 6.2, 6.3, 6.4, 7.0, 7.1, 7.2],
    ICPX: ["2025.0"],
    UBUNTU: [18.04, 20.04, 22.04, 24.04],
    CMAKE: [
        3.19,
        3.20,
        3.21,
        3.22,
        3.23,
        3.24,
        3.25,
        3.26,
        3.27,
        3.28,
        3.29,
        3.30,
    ],
    BOOST: [
        "1.74.0",
        "1.75.0",
        "1.76.0",
        "1.77.0",
        "1.78.0",
        "1.79.0",
        "1.80.0",
        "1.81.0",
        "1.82.0",
        "1.83.0",
        "1.84.0",
        "1.85.0",
        "1.86.0",
    ],
    CXX_STANDARD: [17, 20, 23],
}
# Clang and Clang-CUDA has the same version numbers
VERSIONS[CLANG_CUDA] = copy.copy(VERSIONS[CLANG])


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
                    compilers.append(ParameterValue(sw_name, pkv.parse(str(sw_version))))
        if len(compilers) > 0:
            param_val_matrix[compiler_type] = compilers

    for backend in backends:
        if backend == ALPAKA_ACC_GPU_CUDA_ENABLE:
            param_val_matrix[backend] = [ParameterValue(backend, OFF_VER)]
            for cuda_version in software_versions[NVCC]:
                param_val_matrix[backend].append(
                    ParameterValue(backend, pkv.parse(str(cuda_version)))
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
                param_val_matrix[other].append(ParameterValue(other, pkv.parse(str(version))))

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
        parsed_version = pkv.parse(str(ver))
        # in case of CMAKE, we don't care about the patch level
        if name == CMAKE:
            parsed_version = pkv.parse(f"{parsed_version.major}.{parsed_version.minor}")
            version = pkv.parse(f"{version.major}.{version.minor}")
        if parsed_version == version:
            return True

    return False
