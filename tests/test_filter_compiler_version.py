# pylint: disable=missing-docstring
import unittest

import io
from collections import OrderedDict as OD
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_compiler_version import compiler_version_filter_typechecked


class TestEmptyRow(unittest.TestCase):
    def test_empty_row_shall_always_pass(self):
        self.assertTrue(compiler_version_filter_typechecked(OD()))


class TestHostDeviceCompilerSameVersion(unittest.TestCase):
    def test_valid_combination_rule_v1(self):
        for comb in [
            (ppv((GCC, 10)), ppv((GCC, 10))),
            (ppv((ICPX, "2040.1.0")), ppv((ICPX, "2040.1.0"))),
            (ppv((HIPCC, 5.5)), ppv((HIPCC, 5.5))),
            (ppv((CLANG, 13)), ppv((CLANG, 13))),
            (ppv((CLANG_CUDA, 17)), ppv((CLANG_CUDA, 17))),
        ]:
            self.assertTrue(
                compiler_version_filter_typechecked(
                    OD({HOST_COMPILER: comb[0], DEVICE_COMPILER: comb[1]})
                ),
                f"host compiler and device compiler version are not the same: {comb[0]} != {comb[1]}",
            )

        self.assertTrue(
            compiler_version_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG_CUDA, 10)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                        DEVICE_COMPILER: ppv((CLANG_CUDA, 10)),
                        CMAKE: ppv((CMAKE, 3.18)),
                    }
                )
            ),
        )

        self.assertTrue(
            compiler_version_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG, 14)),
                        DEVICE_COMPILER: ppv((CLANG, 14)),
                        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: ppv(
                            (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, "1.0.0")
                        ),
                        CMAKE: ppv((CMAKE, 3.24)),
                        BOOST: ppv((BOOST, 1.78)),
                    }
                )
            ),
        )

    def test_invalid_combination_rule_v1(self):
        for comb in [
            (ppv((GCC, 10)), ppv((GCC, 11))),
            (ppv((ICPX, "2023.1.0")), ppv((ICPX, "2040.1.0"))),
            (ppv((HIPCC, 6)), ppv((HIPCC, 5.5))),
            (ppv((CLANG, 0)), ppv((CLANG, 13))),
            (ppv((CLANG_CUDA, "4a3")), ppv((CLANG_CUDA, 17))),
        ]:
            reason_msg = io.StringIO()

            self.assertFalse(
                compiler_version_filter_typechecked(
                    OD({HOST_COMPILER: comb[0], DEVICE_COMPILER: comb[1]}), reason_msg
                ),
                f"same host compiler and device compiler version should pass: {comb[0]} and {comb[1]}",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                "host and device compiler version must be the same (except for nvcc)",
            )

        reason_msg_multi1 = io.StringIO()
        self.assertFalse(
            compiler_version_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG_CUDA, 10)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                        DEVICE_COMPILER: ppv((CLANG_CUDA, 11)),
                        CMAKE: ppv((CMAKE, 3.18)),
                    },
                ),
                reason_msg_multi1,
            ),
        )
        self.assertEqual(
            reason_msg_multi1.getvalue(),
            "host and device compiler version must be the same (except for nvcc)",
        )

        reason_msg_multi2 = io.StringIO()
        self.assertFalse(
            compiler_version_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 15)),
                        DEVICE_COMPILER: ppv((GCC, 10)),
                        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: ppv(
                            (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, "1.0.0")
                        ),
                        CMAKE: ppv((CMAKE, 3.24)),
                        BOOST: ppv((BOOST, 1.78)),
                    }
                ),
                reason_msg_multi2,
            ),
        )
        self.assertEqual(
            reason_msg_multi2.getvalue(),
            "host and device compiler version must be the same (except for nvcc)",
        )
