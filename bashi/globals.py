"""This module contains constants used in the bashi library."""

# index positions of the parameter-value
NAME: int = 0
VERSION: int = 1

# parameter key names, whit special meaning
HOST_COMPILER: str = "host_compiler"
DEVICE_COMPILER: str = "device_compiler"

# name of the used compilers
GCC: str = "gcc"
CLANG: str = "clang"
NVCC: str = "nvcc"
CLANG_CUDA: str = "clang-cuda"
HIPCC: str = "hipcc"
ICPX: str = "icpx"

# alpaka backend names
ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: str = "alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE"
ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE: str = "alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE"
ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE: str = "alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE"
ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: str = "alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE"
ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE: str = "alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE"
ALPAKA_ACC_GPU_CUDA_ENABLE: str = "alpaka_ACC_GPU_CUDA_ENABLE"
ALPAKA_ACC_GPU_HIP_ENABLE: str = "alpaka_ACC_GPU_HIP_ENABLE"
ALPAKA_ACC_SYCL_ENABLE: str = "alpaka_ACC_SYCL_ENABLE"

# software dependencies and compiler configurations
UBUNTU: str = "ubuntu"
CMAKE: str = "cmake"
BOOST: str = "boost"
CXX_STANDARD: str = "cxx_standard"
