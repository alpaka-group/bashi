# pylint: disable=missing-docstring
import unittest
import io

from collections import OrderedDict as OD
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_software_dependency import software_dependency_filter_typechecked


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

    def test_valid_ROCm_images_Ubuntu2004_based_d3(self):
        for UBUNTU_version in ["20.04", "22.04", "21.04"]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                            UBUNTU: ppv((UBUNTU, UBUNTU_version)),
                        }
                    ),
                )
            )

        for UBUNTU_version in ["16.04", "18.04", "19.04", "20.04", "22.04", "21.04"]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                            UBUNTU: ppv((UBUNTU, UBUNTU_version)),
                        }
                    ),
                ),
            )
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, 1)),
                            DEVICE_COMPILER: ppv((HIPCC, 1)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                ),
            )

    def test_non_valid_ROCm_images_Ubuntu2004_based_d3(self):
        for UBUNTU_version in ["16.04", "18.04", "19.04"]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                            UBUNTU: ppv((UBUNTU, UBUNTU_version)),
                        }
                    ),
                    reason_msg,
                ),
                f"ROCm and also the hipcc compiler is not available on Ubuntu older than 20.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"ROCm and also the hipcc compiler is not available on Ubuntu older than 20.04",
            )

        for host_name, device_name, hip_backend, ubuntu_version, error_msg in [
            (
                HIPCC,
                HIPCC,
                ON,
                "18.04",
                "ROCm and also the hipcc compiler is not available on Ubuntu older than 20.04",
            ),
            (
                HIPCC,
                GCC,
                ON,
                "18.04",
                "ROCm and also the hipcc compiler is not available on Ubuntu older than 20.04",
            ),
            (
                CLANG,
                HIPCC,
                ON,
                "18.04",
                "ROCm and also the hipcc compiler is not available on Ubuntu older than 20.04",
            ),
            (
                GCC,
                HIPCC,
                ON,
                "18.04",
                "ROCm and also the hipcc compiler is not available on Ubuntu older than 20.04",
            ),
            (
                HIPCC,
                CLANG,
                ON,
                "18.04",
                "ROCm and also the hipcc compiler is not available on Ubuntu older than 20.04",
            ),
        ]:
            test_row = OD(
                {
                    HOST_COMPILER: ppv((host_name, 1)),
                    DEVICE_COMPILER: ppv((device_name, 1)),
                    ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, hip_backend)),
                    UBUNTU: ppv((UBUNTU, ubuntu_version)),
                },
            )
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(test_row, reason_msg),
                f"{test_row}",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                error_msg,
                f"{test_row}",
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

    def test_valid_gcc_cxx_combinations_d5(self):
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 3)),
                        CXX_STANDARD: ppv((CXX_STANDARD, 16)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 7)),
                        CXX_STANDARD: ppv((CXX_STANDARD, 16)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 8)),
                        CXX_STANDARD: ppv((CXX_STANDARD, 16)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 8)),
                        CXX_STANDARD: ppv((CXX_STANDARD, 15)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 9)),
                        CXX_STANDARD: ppv((CXX_STANDARD, 4)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 9)),
                        CXX_STANDARD: ppv((CXX_STANDARD, 8)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 9)),
                        CXX_STANDARD: ppv((CXX_STANDARD, 16)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 10)),
                        CXX_STANDARD: ppv((CXX_STANDARD, 18)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 10)),
                        CXX_STANDARD: ppv((CXX_STANDARD, 19)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, "10.0")),
                        CXX_STANDARD: ppv((CXX_STANDARD, 15)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, "10.2")),
                        CXX_STANDARD: ppv((CXX_STANDARD, 15)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, "10.5")),
                        CXX_STANDARD: ppv((CXX_STANDARD, 15)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, "11.0")),
                        CXX_STANDARD: ppv((CXX_STANDARD, 16)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, "11.0")),
                        CXX_STANDARD: ppv((CXX_STANDARD, 15)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, "11.5")),
                        CXX_STANDARD: ppv((CXX_STANDARD, 15)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, "12.0")),
                        CXX_STANDARD: ppv((CXX_STANDARD, 18)),
                    }
                )
            )
        )
        self.assertTrue(
            software_dependency_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, "12.0")),
                        CXX_STANDARD: ppv((CXX_STANDARD, 19)),
                    }
                )
            )
        )

    def test_invalid_gcc_cxx_combinations_d5(self):
        for cxx_version in [17, 20, 25]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((GCC, 8)),
                            CXX_STANDARD: ppv((CXX_STANDARD, cxx_version)),
                        }
                    ),
                    reason_msg,
                ),
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"host compiler gcc 8 does not support cxx {cxx_version}",
            )
        for cxx_version in [20, 21, 28]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((GCC, 10)),
                            CXX_STANDARD: ppv((CXX_STANDARD, cxx_version)),
                        }
                    ),
                    reason_msg,
                ),
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"host compiler gcc 10 does not support cxx {cxx_version}",
            )

        for cxx_version in [17, 20, 25]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((NVCC, 11.0)),
                            CXX_STANDARD: ppv((CXX_STANDARD, cxx_version)),
                        }
                    ),
                    reason_msg,
                ),
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"device compiler nvcc 11.0 does not support cxx {cxx_version}",
            )

        for cxx_version in [20, 21, 28]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((NVCC, 12.0)),
                            CXX_STANDARD: ppv((CXX_STANDARD, cxx_version)),
                        }
                    ),
                    reason_msg,
                ),
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"device compiler nvcc 12.0 does not support cxx {cxx_version}",
            )
