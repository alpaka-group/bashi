# pylint: disable=missing-docstring
# pylint: disable=too-many-lines
import unittest
import io
from collections import OrderedDict as OD
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_compiler import compiler_filter_typechecked


class TestCompilerCXXSupportFilterRules(unittest.TestCase):
    def test_ignore_combination_gcc_cxx_support_c21(self):
        self.assertTrue(compiler_filter_typechecked(OD({HOST_COMPILER: ppv((GCC, 10))})))
        self.assertTrue(compiler_filter_typechecked(OD({CXX_STANDARD: ppv((CXX_STANDARD, 20))})))
        self.assertTrue(
            compiler_filter_typechecked(
                OD({CXX_STANDARD: ppv((CXX_STANDARD, 20)), CMAKE: ppv((CMAKE, 3.18))})
            )
        )

    def test_valid_in_range_gcc_cxx_support_c21(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 0)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 16)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 16)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 9)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 9)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 10)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 10)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 10)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 11)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 11)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 11)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 9999)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                }
            ),
        ]:
            self.assertTrue(compiler_filter_typechecked(row), f"{row}")

    def test_invalid_in_range_gcc_cxx_support_c21(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 9999)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((GCC, 8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 18)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((GCC, 8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((GCC, 8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((GCC, 9)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 18)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 10)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 21)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 11)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 24)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 11)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 26)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 13)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 26)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 9999)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 26)),
                },
            ),
        ]:
            if HOST_COMPILER in row:
                compiler_type = HOST_COMPILER
            elif DEVICE_COMPILER in row:
                compiler_type = DEVICE_COMPILER
            else:
                compiler_type = ""

            reason_msg = io.StringIO()

            self.assertFalse(compiler_filter_typechecked(row, reason_msg), f"{row}")
            self.assertEqual(
                reason_msg.getvalue(),
                f"{compiler_type} gcc {row[compiler_type].version} does not support "
                f"C++{row[CXX_STANDARD].version}",
            )

    def test_valid_in_range_nvcc_cxx_support_c23(self):
        for row in [
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 10.1)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 0)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 11.0)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 11.0)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 12.3)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 12.8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 12.0)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 99.0)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
        ]:
            self.assertTrue(compiler_filter_typechecked(row), f"{row}")

    def test_invalid_in_range_nvcc_cxx_support_c23(self):
        for row in [
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 10.2)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 9999)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 10.2)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 11.8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 12.9)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                },
            ),
        ]:
            reason_msg = io.StringIO()

            self.assertFalse(compiler_filter_typechecked(row, reason_msg), f"{row}")
            self.assertEqual(
                reason_msg.getvalue(),
                f"{DEVICE_COMPILER} nvcc {row[DEVICE_COMPILER].version} does not support "
                f"C++{row[CXX_STANDARD].version}",
            )
