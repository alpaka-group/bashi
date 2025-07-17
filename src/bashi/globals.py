"""This module contains constants used in the bashi library."""

from typing import List
import packaging.version
from bashi.types import Parameter, ValueName, ValueVersion

# parameter key names, whit special meaning
HOST_COMPILER: Parameter = "host_compiler"
DEVICE_COMPILER: Parameter = "device_compiler"

# name of the used compilers
GCC: ValueName = "gcc"
CLANG: ValueName = "clang"
NVCC: ValueName = "nvcc"
CLANG_CUDA: ValueName = "clang-cuda"
HIPCC: ValueName = "hipcc"
ICPX: ValueName = "icpx"

COMPILERS: List[ValueName] = [GCC, CLANG, NVCC, CLANG_CUDA, HIPCC, ICPX]

# alpaka backend names
ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: str = "alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE"
ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE: str = "alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE"
ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE: str = "alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE"
ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: str = "alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE"
ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE: str = "alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE"
ALPAKA_ACC_GPU_CUDA_ENABLE: str = "alpaka_ACC_GPU_CUDA_ENABLE"
ALPAKA_ACC_GPU_HIP_ENABLE: str = "alpaka_ACC_GPU_HIP_ENABLE"
ALPAKA_ACC_SYCL_ENABLE: str = "alpaka_ACC_SYCL_ENABLE"

BACKENDS: List[str] = [
    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
    ALPAKA_ACC_GPU_CUDA_ENABLE,
    ALPAKA_ACC_GPU_HIP_ENABLE,
    ALPAKA_ACC_SYCL_ENABLE,
]

# software dependencies and compiler configurations
UBUNTU: str = "ubuntu"
CMAKE: str = "cmake"
BOOST: str = "boost"
CXX_STANDARD: str = "cxx_standard"

OFF: str = "0.0.0"
ON: str = "1.0.0"
OFF_VER: ValueVersion = packaging.version.parse(OFF)
ON_VER: ValueVersion = packaging.version.parse(ON)

# values are used for remove_parameter_value_pair
ANY_PARAM: Parameter = "*"
ANY_NAME: ValueName = "*"
ANY_VERSION: str = "*"

# List of all supported parameters
PARAMETERS: List[Parameter] = [
    HOST_COMPILER,
    DEVICE_COMPILER,
    UBUNTU,
    CMAKE,
    BOOST,
    CXX_STANDARD,
] + BACKENDS

# runtime functions
RT_AVAILABLE_HIP_SDK_UBUNTU_VER: str = "rt_available_hip_sdk_ubuntu_ver"
RT_AVAILABLE_CUDA_SDK_UBUNTU_VER: str = "rt_available_cuda_sdk_ubuntu_ver"
