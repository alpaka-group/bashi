# pylint: disable=missing-docstring
import unittest
import io
from collections import OrderedDict as OD
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_compiler import compiler_filter_typechecked
from bashi.filter_backend import backend_filter_typechecked


class TestHipccCompilerFilter(unittest.TestCase):
    def test_hipcc_requires_enabled_hip_backend_pass_c9(self):
        for version in (4.5, 5.3, 6.0):
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    )
                )
            )

    def test_hipcc_requires_enabled_hip_backend_not_pass_c9(self):
        for version in (4.5, 5.3, 6.0):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    ),
                    reason_msg1,
                )
            )
            self.assertEqual(reason_msg1.getvalue(), "hipcc requires an enabled HIP backend.")

            reason_msg2 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    ),
                    reason_msg2,
                )
            )
            self.assertEqual(reason_msg2.getvalue(), "hipcc requires an enabled HIP backend.")

            reason_msg3 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    ),
                    reason_msg3,
                )
            )
            self.assertEqual(reason_msg3.getvalue(), "hipcc requires an enabled HIP backend.")

            reason_msg4 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                        }
                    ),
                    reason_msg4,
                )
            )
            self.assertEqual(reason_msg4.getvalue(), "hipcc requires an enabled HIP backend.")

            reason_msg5 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    ),
                    reason_msg5,
                )
            )
            self.assertEqual(reason_msg5.getvalue(), "hipcc requires an enabled HIP backend.")

    def test_check_if_hip_backend_is_disabled_for_no_hipcc_compiler_pass_b1(self):
        for compiler_name in set(COMPILERS) - set([NVCC, HIPCC]):
            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_GPU_HIP_ENABLE should be off for {compiler_name}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_GPU_HIP_ENABLE should be off for {compiler_name}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_GPU_HIP_ENABLE should be off for {compiler_name}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_GPU_HIP_ENABLE should be off for {compiler_name}",
            )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 9999)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                    }
                )
            ),
            "ALPAKA_ACC_GPU_HIP_ENABLE should be off for nvcc",
        )

        for host_compiler in (GCC, CLANG):
            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((host_compiler, 9999)),
                            DEVICE_COMPILER: ppv((NVCC, 9999)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_GPU_HIP_ENABLE should be off for nvcc + {host_compiler}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((host_compiler, 9999)),
                            DEVICE_COMPILER: ppv((NVCC, 9999)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_GPU_HIP_ENABLE should be off for nvcc + {host_compiler}",
            )

    def test_check_if_hip_backend_is_disabled_for_no_hipcc_compiler_not_pass_b1(self):
        for compiler_name in set(COMPILERS) - set([NVCC, HIPCC]):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    ),
                    reason_msg1,
                ),
                f"{compiler_name} should not pass the filter if ALPAKA_ACC_GPU_HIP_ENABLE is on",
            )
            self.assertEqual(
                reason_msg1.getvalue(), "An enabled HIP backend requires hipcc as compiler."
            )

            reason_msg2 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    ),
                    reason_msg2,
                ),
                f"{compiler_name} should not pass the filter if ALPAKA_ACC_GPU_HIP_ENABLE is on",
            )
            self.assertEqual(
                reason_msg2.getvalue(), "An enabled HIP backend requires hipcc as compiler."
            )

            reason_msg3 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    ),
                    reason_msg3,
                ),
                f"{compiler_name} should not pass the filter if ALPAKA_ACC_GPU_HIP_ENABLE is on",
            )
            self.assertEqual(
                reason_msg3.getvalue(), "An enabled HIP backend requires hipcc as compiler."
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
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    ),
                    reason_msg4,
                ),
                f"{compiler_name} should not pass the filter if ALPAKA_ACC_GPU_HIP_ENABLE is on",
            )
            self.assertEqual(
                reason_msg4.getvalue(), "An enabled HIP backend requires hipcc as compiler."
            )

        reason_msg5 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 9999)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                    }
                ),
                reason_msg5,
            ),
            "nvcc should not pass the filter if ALPAKA_ACC_GPU_HIP_ENABLE is on",
        )
        self.assertEqual(
            reason_msg5.getvalue(), "An enabled HIP backend requires hipcc as compiler."
        )

        for host_compiler in (GCC, CLANG):
            reason_msg6 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((host_compiler, 9999)),
                            DEVICE_COMPILER: ppv((NVCC, 9999)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    ),
                    reason_msg6,
                ),
                f"nvcc + {host_compiler} should not pass the filter if "
                "ALPAKA_ACC_GPU_HIP_ENABLE is on",
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
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    ),
                    reason_msg7,
                ),
                f"nvcc + {host_compiler} should not pass the filter if "
                "ALPAKA_ACC_GPU_HIP_ENABLE is on",
            )

    def test_hipcc_requires_disabled_sycl_backend_pass_c10(self):
        for version in (4.5, 5.3, 6.0):
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        }
                    )
                )
            )

    def test_hipcc_requires_disabled_sycl_backend_not_pass_c10(self):
        for version in (4.5, 5.3, 6.0):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    ),
                    reason_msg1,
                )
            )
            self.assertEqual(reason_msg1.getvalue(), "hipcc does not support the SYCL backend.")

            reason_msg2 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    ),
                    reason_msg2,
                )
            )
            self.assertEqual(reason_msg2.getvalue(), "hipcc does not support the SYCL backend.")

            reason_msg3 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    ),
                    reason_msg3,
                )
            )
            self.assertEqual(reason_msg3.getvalue(), "hipcc does not support the SYCL backend.")

            reason_msg4 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                        }
                    ),
                    reason_msg4,
                )
            )
            self.assertEqual(reason_msg4.getvalue(), "hipcc does not support the SYCL backend.")

            reason_msg5 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        }
                    ),
                    reason_msg5,
                )
            )
            self.assertEqual(reason_msg5.getvalue(), "hipcc does not support the SYCL backend.")

    def test_hip_and_sycl_backend_cannot_be_active_at_the_same_time_b2(self):
        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                    }
                ),
            )
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                    }
                ),
            )
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                    }
                ),
            )
        )

        reason_msg1 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
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

    def test_hipcc_requires_disabled_cuda_backend_pass_c11(self):
        for version in (4.5, 5.3, 6.0):
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
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
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    )
                )
            )

    def test_hipcc_requires_disabled_cuda_backend_not_pass_c11(self):
        for version in (4.5, 5.3, 6.0):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                        }
                    ),
                    reason_msg1,
                )
            )
            self.assertEqual(reason_msg1.getvalue(), "hipcc does not support the CUDA backend.")

            reason_msg2 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                        }
                    ),
                    reason_msg2,
                )
            )
            self.assertEqual(reason_msg2.getvalue(), "hipcc does not support the CUDA backend.")

            reason_msg3 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                        }
                    ),
                    reason_msg3,
                )
            )
            self.assertEqual(reason_msg3.getvalue(), "hipcc does not support the CUDA backend.")

            reason_msg4 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                        }
                    ),
                    reason_msg4,
                )
            )
            self.assertEqual(reason_msg4.getvalue(), "hipcc does not support the CUDA backend.")

            reason_msg5 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                        }
                    ),
                    reason_msg5,
                )
            )
            self.assertEqual(reason_msg5.getvalue(), "hipcc does not support the CUDA backend.")

    def test_hipcc_requires_ubuntu2004_pass_c19(self):
        for version in (4.5, 5.3, 6.0):
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    )
                )
            )
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    )
                )
            )
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            HOST_COMPILER: ppv((HIPCC, version)),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    )
                )
            )
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((GCC, 1)),
                            DEVICE_COMPILER: ppv((GCC, 1)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                )
            )

    def test_hipcc_requires_ubuntu2004_not_pass_c19(self):
        for version in (4.5, 5.3, 6.0):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                    reason_msg1,
                )
            )
            self.assertEqual(
                reason_msg1.getvalue(),
                "ROCm and also the hipcc compiler is not available on Ubuntu older than 20.04",
            )

            reason_msg2 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                    reason_msg2,
                )
            )
            self.assertEqual(
                reason_msg2.getvalue(),
                "ROCm and also the hipcc compiler is not available on Ubuntu older than 20.04",
            )

            reason_msg3 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                    reason_msg3,
                )
            )
            self.assertEqual(
                reason_msg3.getvalue(),
                "ROCm and also the hipcc compiler is not available on Ubuntu older than 20.04",
            )

            reason_msg4 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((HIPCC, version)),
                            DEVICE_COMPILER: ppv((HIPCC, version)),
                            UBUNTU: ppv((UBUNTU, "16.04")),
                        }
                    ),
                    reason_msg4,
                )
            )
            self.assertEqual(
                reason_msg4.getvalue(),
                "ROCm and also the hipcc compiler is not available on Ubuntu older than 20.04",
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
                "host and device compiler name must be the same (except for nvcc)",
            ),
            (
                CLANG,
                HIPCC,
                ON,
                "18.04",
                "host and device compiler name must be the same (except for nvcc)",
            ),
            (
                GCC,
                HIPCC,
                OFF,
                "18.04",
                "host and device compiler name must be the same (except for nvcc)",
            ),
            (
                HIPCC,
                CLANG,
                OFF,
                "18.04",
                "host and device compiler name must be the same (except for nvcc)",
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
                compiler_filter_typechecked(test_row, reason_msg),
                f"{test_row}",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                error_msg,
                f"{test_row}",
            )

    def test_hip_and_cuda_backend_cannot_be_active_at_the_same_time_b3(self):
        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                    }
                ),
            )
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                    }
                ),
            )
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
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
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                    }
                ),
                reason_msg1,
            )
        )
        self.assertEqual(
            reason_msg1.getvalue(), "The HIP and CUDA backend cannot be enabled on the same time."
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        CMAKE: ppv((CMAKE, 3.18)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
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
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        BOOST: ppv((BOOST, "1.78.0")),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
                    }
                ),
                reason_msg2,
            )
        )
        self.assertEqual(
            reason_msg2.getvalue(), "The HIP and CUDA backend cannot be enabled on the same time."
        )
