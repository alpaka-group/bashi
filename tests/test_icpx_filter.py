# pylint: disable=missing-docstring
import unittest
import io
from collections import OrderedDict as OD
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_compiler import compiler_filter_typechecked
from bashi.filter_backend import backend_filter_typechecked


class TestIcpxCompilerFilter(unittest.TestCase):
    def test_icpx_requires_enabled_sycl_backend_pass_c12(self):
        for version in ("2023.3.0", "2024.2.1", "2024.3.0"):
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    )
                )
            )

    def test_icpx_requires_enabled_sycl_backend_not_pass_c12(self):
        for version in ("2023.3.0", "2024.2.1", "2024.3.0"):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    ),
                    reason_msg1,
                )
            )
            self.assertEqual(reason_msg1.getvalue(), "icpx requires an enabled SYCL backend.")

            reason_msg2 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    ),
                    reason_msg2,
                )
            )
            self.assertEqual(reason_msg2.getvalue(), "icpx requires an enabled SYCL backend.")

            reason_msg3 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    ),
                    reason_msg3,
                )
            )
            self.assertEqual(reason_msg3.getvalue(), "icpx requires an enabled SYCL backend.")

            reason_msg4 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                        }
                    ),
                    reason_msg4,
                )
            )
            self.assertEqual(reason_msg4.getvalue(), "icpx requires an enabled SYCL backend.")

            reason_msg5 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    ),
                    reason_msg5,
                )
            )
            self.assertEqual(reason_msg5.getvalue(), "icpx requires an enabled SYCL backend.")

    def test_check_if_sycl_backend_is_disabled_for_no_sycl_compiler_pass_b4(self):
        for compiler_name in set(COMPILERS) - set([NVCC, ICPX]):
            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_SYCL_ENABLE should be off for {compiler_name}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_SYCL_ENABLE should be off for {compiler_name}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_SYCL_ENABLE should be off for {compiler_name}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_SYCL_ENABLE should be off for {compiler_name}",
            )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 9999)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                    }
                )
            ),
            "ALPAKA_ACC_SYCL_ENABLE should be off for nvcc",
        )

        for host_compiler in (GCC, CLANG):
            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((host_compiler, 9999)),
                            DEVICE_COMPILER: ppv((NVCC, 9999)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_SYCL_ENABLE should be off for nvcc + {host_compiler}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((host_compiler, 9999)),
                            DEVICE_COMPILER: ppv((NVCC, 9999)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_SYCL_ENABLE should be off for nvcc + {host_compiler}",
            )

    def test_check_if_sycl_backend_is_disabled_for_no_icpx_compiler_not_pass_b4(self):
        for compiler_name in set(COMPILERS) - set([NVCC, ICPX]):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    ),
                    reason_msg1,
                ),
                f"{compiler_name} should not pass the filter if ALPAKA_ACC_SYCL_ENABLE is on",
            )
            self.assertEqual(
                reason_msg1.getvalue(), "An enabled SYCL backend requires icpx as compiler."
            )

            reason_msg2 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    ),
                    reason_msg2,
                ),
                f"{compiler_name} should not pass the filter if ALPAKA_ACC_SYCL_ENABLE is on",
            )
            self.assertEqual(
                reason_msg2.getvalue(), "An enabled SYCL backend requires icpx as compiler."
            )

            reason_msg3 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    ),
                    reason_msg3,
                ),
                f"{compiler_name} should not pass the filter if ALPAKA_ACC_SYCL_ENABLE is on",
            )
            self.assertEqual(
                reason_msg3.getvalue(), "An enabled SYCL backend requires icpx as compiler."
            )

            reason_msg4 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    ),
                    reason_msg4,
                ),
                f"{compiler_name} should not pass the filter if ALPAKA_ACC_SYCL_ENABLE is on",
            )
            self.assertEqual(
                reason_msg4.getvalue(), "An enabled SYCL backend requires icpx as compiler."
            )

        reason_msg5 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 9999)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                    }
                ),
                reason_msg5,
            ),
            "nvcc should not pass the filter if ALPAKA_ACC_SYCL_ENABLE is on",
        )
        self.assertEqual(
            reason_msg5.getvalue(), "An enabled SYCL backend requires icpx as compiler."
        )

        for host_compiler in (GCC, CLANG):
            reason_msg6 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((host_compiler, 9999)),
                            DEVICE_COMPILER: ppv((NVCC, 9999)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    ),
                    reason_msg6,
                ),
                f"nvcc + {host_compiler} should not pass the filter if "
                "ALPAKA_ACC_SYCL_ENABLE is on",
            )

            reason_msg7 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((host_compiler, 9999)),
                            DEVICE_COMPILER: ppv((NVCC, 9999)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    ),
                    reason_msg7,
                ),
                f"nvcc + {host_compiler} should not pass the filter if "
                "ALPAKA_ACC_SYCL_ENABLE is on",
            )

    def test_sycl_requires_disabled_hip_backend_pass_c13(self):
        for version in ("2023.3.0", "2024.2.1", "2024.3.0"):
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    )
                )
            )

    def test_icpx_requires_disabled_hip_backend_not_pass_c13(self):
        for version in ("2023.3.0", "2024.2.1", "2024.3.0"):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    ),
                    reason_msg1,
                )
            )
            self.assertEqual(reason_msg1.getvalue(), "icpx does not support the HIP backend.")

            reason_msg2 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    ),
                    reason_msg2,
                )
            )
            self.assertEqual(reason_msg2.getvalue(), "icpx does not support the HIP backend.")

            reason_msg3 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    ),
                    reason_msg3,
                )
            )
            self.assertEqual(reason_msg3.getvalue(), "icpx does not support the HIP backend.")

            reason_msg4 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                        }
                    ),
                    reason_msg4,
                )
            )
            self.assertEqual(reason_msg4.getvalue(), "icpx does not support the HIP backend.")

            reason_msg5 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    ),
                    reason_msg5,
                )
            )
            self.assertEqual(reason_msg5.getvalue(), "icpx does not support the HIP backend.")

    def test_sycl_and_hip_backend_cannot_be_active_at_the_same_time_b5(self):
        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                    }
                ),
            )
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                    }
                ),
            )
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                    }
                ),
            )
        )

        reason_msg1 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                    }
                ),
                reason_msg1,
            )
        )
        self.assertEqual(
            reason_msg1.getvalue(), "The HIP and SYCL backend cannot be enabled on the same time."
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        CMAKE: ppv((CMAKE, 3.18)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        BOOST: ppv((BOOST, "1.78.0")),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                    }
                ),
            )
        )

        reason_msg2 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        CMAKE: ppv((CMAKE, 3.18)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        BOOST: ppv((BOOST, "1.78.0")),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                    }
                ),
                reason_msg2,
            )
        )
        self.assertEqual(
            reason_msg2.getvalue(), "The HIP and SYCL backend cannot be enabled on the same time."
        )

    def test_icpx_requires_disabled_cuda_backend_pass_c14(self):
        for version in ("2023.3.0", "2024.2.1", "2024.3.0"):
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    )
                )
            )

    def test_icpx_requires_disabled_cuda_backend_not_pass_c14(self):
        for version in ("2023.3.0", "2024.2.1", "2024.3.0"):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                        }
                    ),
                    reason_msg1,
                )
            )
            self.assertEqual(reason_msg1.getvalue(), "icpx does not support the CUDA backend.")

            reason_msg2 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                        }
                    ),
                    reason_msg2,
                )
            )
            self.assertEqual(reason_msg2.getvalue(), "icpx does not support the CUDA backend.")

            reason_msg3 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                        }
                    ),
                    reason_msg3,
                )
            )
            self.assertEqual(reason_msg3.getvalue(), "icpx does not support the CUDA backend.")

            reason_msg4 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                        }
                    ),
                    reason_msg4,
                )
            )
            self.assertEqual(reason_msg4.getvalue(), "icpx does not support the CUDA backend.")

            reason_msg5 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((ICPX, version)),
                            DEVICE_COMPILER: ppv((ICPX, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                        }
                    ),
                    reason_msg5,
                )
            )
            self.assertEqual(reason_msg5.getvalue(), "icpx does not support the CUDA backend.")

    def test_sycl_and_cuda_backend_cannot_be_active_at_the_same_time_b6(self):
        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                    }
                ),
            )
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                    }
                ),
            )
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                    }
                ),
            )
        )

        reason_msg1 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                    }
                ),
                reason_msg1,
            )
        )
        self.assertEqual(
            reason_msg1.getvalue(), "The SYCL and CUDA backend cannot be enabled on the same time."
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        CMAKE: ppv((CMAKE, 3.18)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        BOOST: ppv((BOOST, "1.78.0")),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                    }
                ),
            )
        )

        reason_msg2 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        CMAKE: ppv((CMAKE, 3.18)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        BOOST: ppv((BOOST, "1.78.0")),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                    }
                ),
                reason_msg2,
            )
        )
        self.assertEqual(
            reason_msg2.getvalue(), "The SYCL and CUDA backend cannot be enabled on the same time."
        )
