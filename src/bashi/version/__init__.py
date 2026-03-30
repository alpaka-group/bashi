"""Provides all supported software versions"""

from typing import Dict, List, Union
import copy
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

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
