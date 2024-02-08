# pylint: disable=missing-docstring
import unittest
import io

from collections import OrderedDict as OD
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_compiler_name import compiler_name_filter


class TestNvccHostCompilerFilter(unittest.TestCase):
    def test_valid_combination_rule_n1(self):
        self.assertTrue(
            compiler_name_filter(
                OD({HOST_COMPILER: ppv((GCC, 10)), DEVICE_COMPILER: ppv((NVCC, 11.2))})
            )
        )

        # version should not matter
        self.assertTrue(
            compiler_name_filter(
                OD({HOST_COMPILER: ppv((CLANG, 0)), DEVICE_COMPILER: ppv((NVCC, 0))})
            )
        )

        self.assertTrue(
            compiler_name_filter(
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
            compiler_name_filter(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 0)),
                        CMAKE: ppv((CMAKE, "3.23")),
                        BOOST: ppv((BOOST, "1.81")),
                    }
                )
            )
        )

        self.assertTrue(compiler_name_filter(OD()))

    def test_invalid_combination_rule_n1(self):
        self.assertFalse(
            compiler_name_filter(
                OD({HOST_COMPILER: ppv((NVCC, 11.2)), DEVICE_COMPILER: ppv((NVCC, 11.2))})
            )
        )

        self.assertFalse(
            compiler_name_filter(
                OD({HOST_COMPILER: ppv((NVCC, 11.2)), DEVICE_COMPILER: ppv((GCC, 11))})
            )
        )

        self.assertFalse(
            compiler_name_filter(
                OD({HOST_COMPILER: ppv((NVCC, 12.2)), DEVICE_COMPILER: ppv((HIPCC, 5.1))})
            )
        )

        self.assertFalse(compiler_name_filter(OD({HOST_COMPILER: ppv((NVCC, 10.2))})))

    def test_reason_rule_n1(self):
        reason_msg = io.StringIO()
        self.assertFalse(compiler_name_filter(OD({HOST_COMPILER: ppv((NVCC, 10.2))}), reason_msg))
        self.assertEqual(reason_msg.getvalue(), "nvcc is not allowed as host compiler")
