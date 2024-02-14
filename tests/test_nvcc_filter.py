# pylint: disable=missing-docstring
import unittest
import io

from collections import OrderedDict as OD
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_compiler_name import compiler_name_filter_typechecked


class TestNoNvccHostCompiler(unittest.TestCase):
    def test_valid_combination_rule_n1(self):
        self.assertTrue(
            compiler_name_filter_typechecked(
                OD({HOST_COMPILER: ppv((GCC, 10)), DEVICE_COMPILER: ppv((NVCC, 11.2))})
            )
        )

        # version should not matter
        self.assertTrue(
            compiler_name_filter_typechecked(
                OD({HOST_COMPILER: ppv((CLANG, 0)), DEVICE_COMPILER: ppv((NVCC, 0))})
            )
        )

        self.assertTrue(
            compiler_name_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG, 0)),
                        DEVICE_COMPILER: ppv((NVCC, 0)),
                        CMAKE: ppv((CMAKE, "3.23")),
                        BOOST: ppv((BOOST, "1.81")),
                    }
                )
            )
        )

        # if HOST_COMPILER does not exist in the row, it should pass because HOST_COMPILER can be
        # added at the next round
        self.assertTrue(
            compiler_name_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 0)),
                        CMAKE: ppv((CMAKE, "3.23")),
                        BOOST: ppv((BOOST, "1.81")),
                    }
                )
            )
        )

    def test_invalid_combination_rule_n1(self):
        reason_msg1 = io.StringIO()
        self.assertFalse(
            compiler_name_filter_typechecked(
                OD({HOST_COMPILER: ppv((NVCC, 11.2)), DEVICE_COMPILER: ppv((NVCC, 11.2))}),
                reason_msg1,
            )
        )
        self.assertEqual(reason_msg1.getvalue(), "nvcc is not allowed as host compiler")

        reason_msg2 = io.StringIO()
        self.assertFalse(
            compiler_name_filter_typechecked(
                OD({HOST_COMPILER: ppv((NVCC, 11.2)), DEVICE_COMPILER: ppv((GCC, 11))}), reason_msg2
            )
        )
        self.assertEqual(reason_msg2.getvalue(), "nvcc is not allowed as host compiler")

        reason_msg3 = io.StringIO()
        self.assertFalse(
            compiler_name_filter_typechecked(
                OD({HOST_COMPILER: ppv((NVCC, 12.2)), DEVICE_COMPILER: ppv((HIPCC, 5.1))}),
                reason_msg3,
            )
        )
        self.assertEqual(reason_msg3.getvalue(), "nvcc is not allowed as host compiler")

        reason_msg4 = io.StringIO()
        self.assertFalse(
            compiler_name_filter_typechecked(OD({HOST_COMPILER: ppv((NVCC, 10.2))}), reason_msg4)
        )
        self.assertEqual(reason_msg4.getvalue(), "nvcc is not allowed as host compiler")


class TestSupportedNvccHostCompiler(unittest.TestCase):
    def test_invalid_combination_rule_n2(self):
        for compiler_name in [CLANG_CUDA, HIPCC, ICPX, NVCC]:
            for compiler_version in ["0", "13", "32a2"]:
                reason_msg = io.StringIO()
                self.assertFalse(
                    compiler_name_filter_typechecked(
                        OD(
                            {
                                HOST_COMPILER: ppv((compiler_name, compiler_version)),
                                DEVICE_COMPILER: ppv((NVCC, "12.3")),
                            }
                        ),
                        reason_msg,
                    )
                )
                # NVCC is filtered by rule n1
                if compiler_name != NVCC:
                    self.assertEqual(
                        reason_msg.getvalue(),
                        "only gcc and clang are allowed as nvcc host compiler",
                    )

        reason_msg1 = io.StringIO()
        self.assertFalse(
            compiler_name_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((HIPCC, "5.3")),
                        DEVICE_COMPILER: ppv((NVCC, "12.3")),
                        CMAKE: ppv((CMAKE, "3.18")),
                        BOOST: ppv((BOOST, "1.81.0")),
                    }
                ),
                reason_msg1,
            )
        )
        self.assertEqual(
            reason_msg1.getvalue(),
            "only gcc and clang are allowed as nvcc host compiler",
        )

        reason_msg2 = io.StringIO()
        self.assertFalse(
            compiler_name_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((HIPCC, "5.3")),
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: ppv(
                            (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON)
                        ),
                        DEVICE_COMPILER: ppv((NVCC, "12.3")),
                    }
                ),
                reason_msg2,
            )
        )
        self.assertEqual(
            reason_msg2.getvalue(),
            "only gcc and clang are allowed as nvcc host compiler",
        )

    def test_valid_combination_rule_n2(self):
        for compiler_name in [GCC, CLANG]:
            for compiler_version in ["0", "13", "7b2"]:
                self.assertTrue(
                    compiler_name_filter_typechecked(
                        OD(
                            {
                                HOST_COMPILER: ppv((compiler_name, compiler_version)),
                                DEVICE_COMPILER: ppv((NVCC, "12.3")),
                            }
                        )
                    )
                )

        self.assertTrue(
            compiler_name_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, "13")),
                        DEVICE_COMPILER: ppv((NVCC, "11.5")),
                        BOOST: ppv((BOOST, "1.84.0")),
                        CMAKE: ppv((CMAKE, "3.23")),
                    }
                )
            )
        )
        self.assertTrue(
            compiler_name_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG, "14")),
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: ppv(
                            (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON)
                        ),
                        DEVICE_COMPILER: ppv((NVCC, "10.1")),
                    }
                )
            )
        )
