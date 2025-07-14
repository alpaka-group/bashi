# pylint: disable=missing-docstring
import unittest
import io

from typing import Dict, Callable
from collections import OrderedDict as OD
from utils_test import parse_param_val as ppv
from utils_test import parse_value_version
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_software_dependency import (
    software_dependency_filter_typechecked,
    SoftwareDependencyFilter,
)
from bashi.runtime_info import ValidUbuntuHip


class TestOldGCCVersionInUbuntu2004(unittest.TestCase):
    def test_valid_gcc_is_in_ubuntu_2004_d1(self):
        for gcc_version in [7, 13, 99]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD({HOST_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "22.04"))}),
                )
            )

            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((GCC, gcc_version)),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    ),
                )
            )
        for gcc_version in [7, 13, 99]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD({HOST_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "20.04"))}),
                )
            )

            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((GCC, gcc_version)),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    ),
                )
            )
        for gcc_version in [1, 3, 6, 7, 13, 99]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD({HOST_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "18.04"))}),
                )
            )
        for gcc_version in [1, 3, 6, 7, 13, 99]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD({DEVICE_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "18.04"))}),
                )
            )
        for gcc_version in [1, 3, 6, 7, 13, 99]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((GCC, gcc_version)),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                )
            )
        for gcc_version in [1, 3, 6, 7, 13, 99]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((GCC, gcc_version)),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                )
            )

    def test_not_valid_gcc_is_in_ubuntu_2004_d1(self):
        for gcc_version in [6, 3, 1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD({HOST_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "20.04"))}),
                    reason_msg,
                ),
                f"host compiler GCC {gcc_version} + Ubuntu 20.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"host compiler GCC {gcc_version} is not available in Ubuntu 20.04",
            )
        for gcc_version in [6, 3, 1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD({HOST_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "22.04"))}),
                    reason_msg,
                ),
                f"host compiler GCC {gcc_version} + Ubuntu 22.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"host compiler GCC {gcc_version} is not available in Ubuntu 22.04",
            )

        for gcc_version in [6, 3, 1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD({DEVICE_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "20.04"))}),
                    reason_msg,
                ),
                f"device compiler GCC {gcc_version} + Ubuntu 20.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"device compiler GCC {gcc_version} is not available in Ubuntu 20.04",
            )
        for gcc_version in [6, 3, 1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD({DEVICE_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "22.04"))}),
                    reason_msg,
                ),
                f"device compiler GCC {gcc_version} + Ubuntu 22.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"device compiler GCC {gcc_version} is not available in Ubuntu 22.04",
            )

    def test_valid_cmake_versions_for_clangcuda_d2(self):
        for cmake_version in ["3.19", "3.26", "3.20", "3.49"]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, 14)),
                            CMAKE: ppv((CMAKE, cmake_version)),
                        }
                    ),
                )
            )

        for cmake_version in ["3.19", "3.26", "3.20", "3.49"]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((CLANG_CUDA, 15)),
                            CMAKE: ppv((CMAKE, cmake_version)),
                        }
                    ),
                )
            )

    def test_not_valid_cmake_versions_for_clangcuda_d2(self):
        for cmake_version in ["3.9", "3.11", "3.17", "3.18"]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD({HOST_COMPILER: ppv((CLANG_CUDA, 14)), CMAKE: ppv((CMAKE, cmake_version))}),
                    reason_msg,
                ),
                f"host compiler CLANG_CUDA + CMAKE {cmake_version}",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"host compiler CLANG_CUDA is not available in CMAKE {cmake_version}",
            )

        for cmake_version in ["3.9", "3.11", "3.17", "3.18"]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {DEVICE_COMPILER: ppv((CLANG_CUDA, 15)), CMAKE: ppv((CMAKE, cmake_version))}
                    ),
                    reason_msg,
                ),
                f"device compiler CLANG_CUDA + CMAKE {cmake_version}",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"device compiler CLANG_CUDA is not available in CMAKE {cmake_version}",
            )

    def test_valid_cuda_versions_for_ubuntu_d4(self):
        for CUDA_version in [11.1, 11.4, 15]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, CUDA_version)
                            ),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    ),
                )
            )

        for CUDA_version in [11.2, 11.4, 15]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, CUDA_version)
                            ),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    ),
                )
            )
        for CUDA_version in [2, 6, 10.1, 10.2, 11.4, 15]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, CUDA_version)
                            ),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                )
            )

        for nvcc_version in [2, 6, 10.1, 10.2, 11.4, 15]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((NVCC, nvcc_version)),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                )
            )
        for nvcc_version in [11.1, 11.2, 11.4, 15]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((NVCC, nvcc_version)),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    ),
                )
            )
        for nvcc_version in [11.1, 11.2, 11.4, 15]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((NVCC, nvcc_version)),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    ),
                )
            )
        for clang_cuda_version in [2, 6, 10.1, 10.2, 11, 12.4, 15]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                )
            )
        for clang_cuda_version in [12, 15, 27]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    ),
                )
            )
        for clang_cuda_version in [12, 15, 27]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    ),
                )
            )
        for clang_cuda_version in [2, 6, 10.1, 10.2, 11, 12.4, 15]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                )
            )
        for clang_cuda_version in [12, 15, 27]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    ),
                )
            )
        for clang_cuda_version in [12, 17, 27]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    ),
                )
            )

    def test_not_valid_cuda_versions_for_ubuntu_d4(self):
        for CUDA_version in [1, 6, 10.1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, CUDA_version)
                            ),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    ),
                    reason_msg,
                ),
                f"CUDA {CUDA_version} is not available in Ubuntu 20.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"CUDA {CUDA_version} is not available in Ubuntu 20.04",
            )
        for CUDA_version in [1, 6, 10.1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, CUDA_version)
                            ),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    ),
                    reason_msg,
                ),
                f"CUDA {CUDA_version} is not available in Ubuntu 22.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"CUDA {CUDA_version} is not available in Ubuntu 22.04",
            )
        for nvcc_version in [1, 6, 10.1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((NVCC, nvcc_version)),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    ),
                    reason_msg,
                ),
                f"NVCC {nvcc_version} is not available in Ubuntu 20.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"NVCC {nvcc_version} is not available in Ubuntu 20.04",
            )
        for nvcc_version in [1, 6, 10.1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((NVCC, nvcc_version)),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    ),
                    reason_msg,
                ),
                f"NVCC {nvcc_version} is not available in Ubuntu 22.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"NVCC {nvcc_version} is not available in Ubuntu 22.04",
            )

        for clang_cuda_version in [1, 6, 10.1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    ),
                    reason_msg,
                ),
                f"device compiler clang-cuda {clang_cuda_version} is not available in Ubuntu 22.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"device compiler clang-cuda {clang_cuda_version} is not available in Ubuntu 22.04",
            )
        for clang_cuda_version in [1, 6, 10.1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    ),
                    reason_msg,
                ),
                f"host compiler clang-cuda {clang_cuda_version} is not available in Ubuntu 22.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"host compiler clang-cuda {clang_cuda_version} is not available in Ubuntu 22.04",
            )

        for clang_cuda_version in [1, 6, 10.1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    ),
                    reason_msg,
                ),
                f"device compiler clang-cuda {clang_cuda_version} is not available in Ubuntu 20.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"device compiler clang-cuda {clang_cuda_version} is not available in Ubuntu 20.04",
            )
        for clang_cuda_version in [1, 6, 10.1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    ),
                    reason_msg,
                ),
                f"host compiler clang-cuda {clang_cuda_version} is not available in Ubuntu 20.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"host compiler clang-cuda {clang_cuda_version} is not available in Ubuntu 20.04",
            )


class TestHIPUbuntu(unittest.TestCase):
    def test_valid_hipcc_ubuntu_d3(self):
        for ubuntu_ver, hipcc_ver in [
            ("20.04", 5.0),
            ("20.04", 5.3),
            ("20.04", 5.9),
            ("22.04", 6.0),
            ("22.04", 6.2),
            ("24.04", 6.3),
        ]:
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                self.assertTrue(
                    software_dependency_filter_typechecked(
                        OD(
                            {
                                compiler_type: ppv((HIPCC, hipcc_ver)),
                                UBUNTU: ppv((UBUNTU, ubuntu_ver)),
                            }
                        ),
                    )
                )

    def test_invalid_hipcc_ubuntu_d3(self):
        for ubuntu_ver, hipcc_ver in [
            ("14.04", 3.0),
            ("18.04", 4.0),
            ("18.04", 5.0),
            ("22.04", 5.0),
            ("20.04", 6.0),
            ("24.04", 6.0),
            ("24.04", 6.2),
        ]:
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                reason_msg = io.StringIO()
                self.assertFalse(
                    software_dependency_filter_typechecked(
                        OD(
                            {
                                compiler_type: ppv((HIPCC, hipcc_ver)),
                                UBUNTU: ppv((UBUNTU, ubuntu_ver)),
                            }
                        ),
                        reason_msg,
                    )
                )

                self.assertEqual(
                    reason_msg.getvalue(),
                    f"The hipcc {hipcc_ver} compiler is not available on "
                    f"the Ubuntu {ubuntu_ver} image.",
                )

    def test_valid_hip_backend_ubuntu_no_runtime_info_d3(self):
        for ubuntu_ver in [
            "20.04",
            "20.04",
            "20.04",
            "22.04",
            "22.04",
            "24.04",
        ]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                            UBUNTU: ppv((UBUNTU, ubuntu_ver)),
                        }
                    ),
                )
            )

    def test_valid_hip_backend_ubuntu_runtime_info_d3(self):
        runtime_info: Dict[str, Callable[..., bool]] = {}
        runtime_info[RT_AVAILABLE_HIP_SDK_UBUNTU_VER] = ValidUbuntuHip(
            parse_value_version(["20.04", "22.04", "26.04"])
        )
        sw_dep_filter = SoftwareDependencyFilter(runtime_infos=runtime_info)

        for ubuntu_ver in [
            "20.04",
            "22.04",
            "26.04",
        ]:
            self.assertTrue(
                sw_dep_filter(
                    OD(
                        {
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                            UBUNTU: ppv((UBUNTU, ubuntu_ver)),
                        }
                    ),
                )
            )

    def test_invalid_hip_backend_ubuntu_runtime_info_d5(self):
        runtime_info: Dict[str, Callable[..., bool]] = {}
        runtime_info[RT_AVAILABLE_HIP_SDK_UBUNTU_VER] = ValidUbuntuHip(
            parse_value_version(["20.04", "22.04", "26.04"])
        )

        for ubuntu_ver in [
            "18.04",
            "24.04",
            "30.04",
        ]:
            reason_msg = io.StringIO()
            sw_dep_filter = SoftwareDependencyFilter(runtime_infos=runtime_info, output=reason_msg)

            self.assertFalse(
                sw_dep_filter(
                    OD(
                        {
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                            UBUNTU: ppv((UBUNTU, ubuntu_ver)),
                        }
                    ),
                )
            )

            self.assertEqual(
                reason_msg.getvalue(),
                f"There is no HIP SDK in input parameter-value-matrix which can be installed on Ubuntu {ubuntu_ver}",
            )
