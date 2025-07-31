# pylint: disable=missing-docstring
import unittest
import io

from typing import List, Tuple
from collections import OrderedDict as OD
import packaging.version as pkv
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import VERSIONS, CLANG_CUDA_MAX_CUDA_VERSION
from bashi.filter_compiler import compiler_filter_typechecked
from bashi.filter_backend import backend_filter_typechecked


class TestClangCUDACompilerFilter(unittest.TestCase):
    def test_clang_cuda_requires_enabled_cuda_backend_c15(self):
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            compiler_type: ppv((CLANG_CUDA, 15)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                        }
                    )
                )
            )

            reason_msg1 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            compiler_type: ppv((CLANG_CUDA, 15)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    ),
                    reason_msg1,
                )
            )
            self.assertEqual(reason_msg1.getvalue(), "clang-cuda requires an enabled CUDA backend.")

    def test_clang_cuda_supported_cuda_backends_c16(self):
        self.assertEqual(
            CLANG_CUDA_MAX_CUDA_VERSION[0].clang_cuda,
            pkv.parse("17"),
            "Modify this test, if a new supported Clang-CUDA version is added. Afterwards change "
            "the last supported version of this assert.",
        )

        # because of rule c8, Clang-CUDA 13 and older is not supported
        clang_cuda_sdk_combinations: List[Tuple[int, float, bool]] = [
            (14, 12.2, False),
            (14, 10.2, True),
            (14, 11.5, True),
            (14, 11.6, False),
            (15, 11.5, True),
            (15, 11.6, False),
            (16, 11.8, True),
            (16, 12.0, False),
            (17, 12.1, True),
            (17, 12.2, False),
        ]

        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            for (
                clang_cuda_version,
                cuda_sdk_version,
                expected_filter_result,
            ) in clang_cuda_sdk_combinations:
                reason_msg1 = io.StringIO()

                self.assertEqual(
                    compiler_filter_typechecked(
                        OD(
                            {
                                compiler_type: ppv((CLANG_CUDA, clang_cuda_version)),
                                ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                    (ALPAKA_ACC_GPU_CUDA_ENABLE, cuda_sdk_version)
                                ),
                            }
                        ),
                        reason_msg1,
                    ),
                    expected_filter_result,
                    f"Clang-CUDA {clang_cuda_version} + CUDA {cuda_sdk_version} -> "
                    f"expected {expected_filter_result}",
                )

                if not expected_filter_result:
                    self.assertEqual(
                        reason_msg1.getvalue(),
                        f"clang-cuda {clang_cuda_version} does not support "
                        f"CUDA {cuda_sdk_version}.",
                    )

    def test_unsupported_new_clang_cuda_version_c16(self):
        unsupported_new_clang_cuda_version = sorted(VERSIONS[CLANG_CUDA])[-1]
        # only verify the following calculation
        self.assertIsInstance(unsupported_new_clang_cuda_version, int)
        unsupported_new_clang_cuda_version = int(unsupported_new_clang_cuda_version) + 1

        unsupported_new_cuda_sdk_version = sorted(VERSIONS[NVCC])[-1]
        # only verify the following calculation
        self.assertIsInstance(unsupported_new_cuda_sdk_version, float)
        unsupported_new_cuda_sdk_version = float(unsupported_new_cuda_sdk_version) + 1.0

        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            compiler_type: ppv((CLANG_CUDA, unsupported_new_clang_cuda_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 9.0)),
                        }
                    ),
                ),
                f"clang-cuda {unsupported_new_clang_cuda_version} + CUDA 9.0",
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            compiler_type: ppv((CLANG_CUDA, unsupported_new_clang_cuda_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, sorted(VERSIONS[NVCC])[-1])
                            ),
                        }
                    ),
                ),
                f"clang-cuda {unsupported_new_clang_cuda_version} + "
                f"CUDA {sorted(VERSIONS[NVCC])[-1]}",
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            compiler_type: ppv((CLANG_CUDA, unsupported_new_clang_cuda_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, unsupported_new_cuda_sdk_version)
                            ),
                        }
                    ),
                ),
                f"clang-cuda {unsupported_new_clang_cuda_version} + "
                f"CUDA {unsupported_new_cuda_sdk_version}",
            )

    def test_clang_cuda_does_not_support_the_hip_backend_c30(self):
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            compiler_type: ppv((CLANG_CUDA, 15)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        }
                    )
                )
            )

            reason_msg1 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            compiler_type: ppv((CLANG_CUDA, 15)),
                            ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        }
                    ),
                    reason_msg1,
                )
            )
            self.assertEqual(reason_msg1.getvalue(), "clang-cuda does not support the HIP backend.")

    def test_valid_clang_cuda_does_not_support_the_sycl_backend_c18(self):
        for row in [
            OD({HOST_COMPILER: ppv((CLANG_CUDA, 15))}),
            OD({DEVICE_COMPILER: ppv((CLANG_CUDA, 14))}),
            OD({HOST_COMPILER: ppv((CLANG_CUDA, 17)), DEVICE_COMPILER: ppv((CLANG_CUDA, 17))}),
        ]:
            for comb in [
                [ALPAKA_ACC_ONEAPI_CPU_ENABLE],
                [ALPAKA_ACC_ONEAPI_GPU_ENABLE],
                [ALPAKA_ACC_ONEAPI_FPGA_ENABLE],
                [ALPAKA_ACC_ONEAPI_CPU_ENABLE, ALPAKA_ACC_ONEAPI_GPU_ENABLE],
                [ALPAKA_ACC_ONEAPI_CPU_ENABLE, ALPAKA_ACC_ONEAPI_FPGA_ENABLE],
                [ALPAKA_ACC_ONEAPI_GPU_ENABLE, ALPAKA_ACC_ONEAPI_FPGA_ENABLE],
                [
                    ALPAKA_ACC_ONEAPI_CPU_ENABLE,
                    ALPAKA_ACC_ONEAPI_GPU_ENABLE,
                    ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
                ],
            ]:
                for backend_name in comb:
                    row[backend_name] = ppv((backend_name, OFF))
                self.assertTrue(compiler_filter_typechecked(row), f"{row}")

    def test_invalid_clang_cuda_does_not_support_the_sycl_backend_c18(self):
        for row in [
            OD({HOST_COMPILER: ppv((CLANG_CUDA, 15))}),
            OD({DEVICE_COMPILER: ppv((CLANG_CUDA, 14))}),
            OD({HOST_COMPILER: ppv((CLANG_CUDA, 17)), DEVICE_COMPILER: ppv((CLANG_CUDA, 17))}),
        ]:
            for comb in [
                [(ALPAKA_ACC_ONEAPI_CPU_ENABLE, ON)],
                [(ALPAKA_ACC_ONEAPI_GPU_ENABLE, ON)],
                [(ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)],
                [(ALPAKA_ACC_ONEAPI_CPU_ENABLE, ON), (ALPAKA_ACC_ONEAPI_GPU_ENABLE, OFF)],
                [(ALPAKA_ACC_ONEAPI_CPU_ENABLE, OFF), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)],
                [(ALPAKA_ACC_ONEAPI_GPU_ENABLE, ON), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)],
                [
                    (ALPAKA_ACC_ONEAPI_CPU_ENABLE, ON),
                    (ALPAKA_ACC_ONEAPI_GPU_ENABLE, OFF),
                    (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF),
                ],
                [
                    (ALPAKA_ACC_ONEAPI_CPU_ENABLE, OFF),
                    (ALPAKA_ACC_ONEAPI_GPU_ENABLE, OFF),
                    (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON),
                ],
                [
                    (ALPAKA_ACC_ONEAPI_CPU_ENABLE, OFF),
                    (ALPAKA_ACC_ONEAPI_GPU_ENABLE, ON),
                    (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON),
                ],
                [
                    (ALPAKA_ACC_ONEAPI_CPU_ENABLE, ON),
                    (ALPAKA_ACC_ONEAPI_GPU_ENABLE, ON),
                    (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON),
                ],
            ]:
                for backend_name, value in comb:
                    row[backend_name] = ppv((backend_name, value))

                reason_msg = io.StringIO()
                self.assertFalse(compiler_filter_typechecked(row, reason_msg), f"{row}")
                self.assertEqual(
                    reason_msg.getvalue(), "clang-cuda does not support the SYCL backend."
                )


class TestClangCUDABackendFilter(unittest.TestCase):
    def test_cuda_backend_supported_clang_cuda_version_b17(self):
        self.assertEqual(
            CLANG_CUDA_MAX_CUDA_VERSION[0].clang_cuda,
            pkv.parse("17"),
            "Modify this test, if a new supported Clang-CUDA version is added. Afterwards change "
            "the last supported version of this assert.",
        )

        # because of rule c8, Clang-CUDA 13 and older is not supported
        clang_cuda_sdk_combinations: List[Tuple[int, float, bool]] = [
            (14, 12.2, False),
            (14, 10.2, True),
            (14, 11.5, True),
            (14, 11.6, False),
            (15, 11.5, True),
            (15, 11.6, False),
            (16, 11.8, True),
            (16, 12.0, False),
            (17, 12.1, True),
            (17, 12.2, False),
        ]

        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            for (
                clang_cuda_version,
                cuda_sdk_version,
                expected_filter_result,
            ) in clang_cuda_sdk_combinations:
                reason_msg1 = io.StringIO()
                self.assertEqual(
                    backend_filter_typechecked(
                        OD(
                            {
                                compiler_type: ppv((CLANG_CUDA, clang_cuda_version)),
                                ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                    (ALPAKA_ACC_GPU_CUDA_ENABLE, cuda_sdk_version)
                                ),
                            }
                        ),
                        reason_msg1,
                    ),
                    expected_filter_result,
                    f"Clang-CUDA {clang_cuda_version} + CUDA {cuda_sdk_version} -> "
                    f"expected {expected_filter_result}",
                )

                if not expected_filter_result:
                    self.assertEqual(
                        reason_msg1.getvalue(),
                        f"CUDA {cuda_sdk_version} is not supported "
                        f"by Clang-CUDA {clang_cuda_version}",
                    )

    def test_unsupported_new_clang_cuda_version_b17(self):
        unsupported_new_clang_cuda_version = sorted(VERSIONS[CLANG_CUDA])[-1]
        # only verify the following calculation
        self.assertIsInstance(unsupported_new_clang_cuda_version, int)
        unsupported_new_clang_cuda_version = int(unsupported_new_clang_cuda_version) + 1

        unsupported_new_cuda_sdk_version = sorted(VERSIONS[NVCC])[-1]
        # only verify the following calculation
        self.assertIsInstance(unsupported_new_cuda_sdk_version, float)
        unsupported_new_cuda_sdk_version = float(unsupported_new_cuda_sdk_version) + 1.0

        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            compiler_type: ppv((CLANG_CUDA, unsupported_new_clang_cuda_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 9.0)),
                        }
                    ),
                ),
                f"clang-cuda {unsupported_new_clang_cuda_version} + CUDA 9.0",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            compiler_type: ppv((CLANG_CUDA, unsupported_new_clang_cuda_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, sorted(VERSIONS[NVCC])[-1])
                            ),
                        }
                    ),
                ),
                f"clang-cuda {unsupported_new_clang_cuda_version} + "
                f"CUDA {sorted(VERSIONS[NVCC])[-1]}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            compiler_type: ppv((CLANG_CUDA, unsupported_new_clang_cuda_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, unsupported_new_cuda_sdk_version)
                            ),
                        }
                    ),
                ),
                f"clang-cuda {unsupported_new_clang_cuda_version} + "
                f"CUDA {unsupported_new_cuda_sdk_version}",
            )
