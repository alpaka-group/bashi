"""Provides all supported software versions"""

from typing import Dict, List, Union
from collections import OrderedDict
import packaging.version as pkv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValue, ParameterValueMatrix

VERSIONS: Dict[str, List[Union[str, int, float]]] = {
    GCC: [6, 7, 8, 9, 10, 11, 12, 13],
    CLANG: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    NVCC: [
        10.0,
        10.1,
        10.2,
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
    ],
    HIPCC: [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 6.0],
    ICPX: ["2023.1.0", "2023.2.0"],
    UBUNTU: [18.04, 20.04],
    CMAKE: [3.18, 3.19, 3.20, 3.21, 3.22, 3.23, 3.24, 3.25, 3.26],
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
    ],
    CXX_STANDARD: [17, 20],
}


def get_parameter_value_matrix() -> ParameterValueMatrix:
    """Generates a parameter-value-matrix from all supported compilers, softwares and compilation
    configuration.

    Returns:
        ParameterValueMatrix: parameter-value-matrix
    """
    param_val_matrix: ParameterValueMatrix = OrderedDict()

    extended_version = VERSIONS.copy()
    extended_version[CLANG_CUDA] = extended_version[CLANG]

    for compiler_type in [HOST_COMPILER, DEVICE_COMPILER]:
        param_val_matrix[compiler_type] = []
        for sw_name, sw_versions in extended_version.items():
            if sw_name in COMPILERS:
                for sw_version in sw_versions:
                    param_val_matrix[compiler_type].append(
                        ParameterValue(sw_name, pkv.parse(str(sw_version)))
                    )

    for backend in BACKENDS:
        if backend == ALPAKA_ACC_GPU_CUDA_ENABLE:
            param_val_matrix[backend] = [ParameterValue(backend, OFF_VER)]
            for cuda_version in extended_version[NVCC]:
                param_val_matrix[backend].append(
                    ParameterValue(backend, pkv.parse(str(cuda_version)))
                )
        else:
            param_val_matrix[backend] = [
                ParameterValue(backend, OFF_VER),
                ParameterValue(backend, ON_VER),
            ]

    for other, versions in extended_version.items():
        if not other in COMPILERS + BACKENDS:
            param_val_matrix[other] = []
            for version in versions:
                param_val_matrix[other].append(ParameterValue(other, pkv.parse(str(version))))

    return param_val_matrix
