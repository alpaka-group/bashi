# pylint: disable=missing-docstring
import unittest

import io
from collections import OrderedDict as OD
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_compiler_name import compiler_name_filter_typechecked


class TestEmptyRow(unittest.TestCase):
    def test_empty_row_shall_always_pass(self):
        self.assertTrue(compiler_name_filter_typechecked(OD()))


class TestHostDeviceCompilerSameName(unittest.TestCase):
    def test_valid_combination_rule_n3(self):
        for comb in [
            (ppv((GCC, 10)), ppv((GCC, 10))),
            (ppv((GCC, 1)), ppv((GCC, 10))),  # version is not important
            (ppv((HIPCC, 6.0)), ppv((HIPCC, 6.0))),
            (ppv((ICPX, 3.0)), ppv((ICPX, 6.0))),
            (ppv((CLANG, 3.0)), ppv((NVCC, 6.0))),  # nvcc has device compiler is an exception
            (ppv((GCC, 3.0)), ppv((NVCC, 6.0))),
        ]:
            self.assertTrue(
                compiler_name_filter_typechecked(
                    OD({HOST_COMPILER: comb[0], DEVICE_COMPILER: comb[1]})
                ),
                f"host compiler and device compiler name are not the same: {comb[0]} != {comb[1]}",
            )

        self.assertTrue(
            compiler_name_filter_typechecked(
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
            compiler_name_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG, 10)),
                        DEVICE_COMPILER: ppv((CLANG, 10)),
                        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: ppv(
                            (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, "1.0.0")
                        ),
                        CMAKE: ppv((CMAKE, 3.24)),
                        BOOST: ppv((BOOST, 1.78)),
                    }
                )
            ),
        )

    def test_invalid_combination_rule_n3(self):
        for comb in [
            (ppv((GCC, 10)), ppv((CLANG, 10))),
            (ppv((HIPCC, 1)), ppv((GCC, 10))),  # version is not important
            (ppv((HIPCC, 6.0)), ppv((ICPX, 6.0))),
            (ppv((ICPX, 3.0)), ppv((CLANG_CUDA, 6.0))),
        ]:
            reason_msg = io.StringIO()

            self.assertFalse(
                compiler_name_filter_typechecked(
                    OD({HOST_COMPILER: comb[0], DEVICE_COMPILER: comb[1]}), reason_msg
                ),
                f"same host compiler and device compiler name should pass: {comb[0]} and {comb[1]}",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                "host and device compiler name must be the same (except for nvcc)",
            )

        reason_msg_multi1 = io.StringIO()
        self.assertFalse(
            compiler_name_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG_CUDA, 10)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                        DEVICE_COMPILER: ppv((HIPCC, 10)),
                        CMAKE: ppv((CMAKE, 3.18)),
                    },
                ),
                reason_msg_multi1,
            ),
        )
        self.assertEqual(
            reason_msg_multi1.getvalue(),
            "host and device compiler name must be the same (except for nvcc)",
        )

        reason_msg_multi2 = io.StringIO()
        self.assertFalse(
            compiler_name_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 15)),
                        DEVICE_COMPILER: ppv((CLANG, 10)),
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
            "host and device compiler name must be the same (except for nvcc)",
        )
