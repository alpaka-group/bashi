# pylint: disable=missing-docstring
# pylint: disable=too-many-lines
import unittest

# pyright: reportPrivateUsage=false
from bashi.result_modules.cxx_compiler_support import _remove_unsupported_cxx_version_for_compiler

# pyright: reportPrivateUsage=false
from bashi.result_modules.cxx_compiler_support import (
    _remove_unsupported_cxx_versions_for_gcc,
    _remove_unsupported_cxx_versions_for_clang,
)
from bashi.types import ParameterValuePair
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import CompilerCxxSupport
from utils_test import (
    parse_expected_val_pairs2,
    default_remove_test,
    create_diff_parameter_value_pairs,
)


class TestCompilerCXXSupportResultFilter(unittest.TestCase):
    def test_older_standard_than_cxx_11(self):
        compiler_support_list = [
            CompilerCxxSupport("8", "11"),
            CompilerCxxSupport("9", "14"),
        ]
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 11)),
                ((HOST_COMPILER, GCC, 9), (CXX_STANDARD, 14)),
            ]
        )
        removed_param_value_pairs: List[ParameterValuePair] = []

        with self.assertRaises(RuntimeError):
            _remove_unsupported_cxx_version_for_compiler(
                test_param_value_pairs,
                removed_param_value_pairs,
                GCC,
                compiler_support_list,
                HOST_COMPILER,
            )

    def test_remove_unsupported_cxx_versions_for_gcc(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                # not affected
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((DEVICE_COMPILER, GCC, 6), (BOOST, "1.81.1")),
                ((UBUNTU, "20.04"), (CXX_STANDARD, 17)),
                ((CXX_STANDARD, 17), (BOOST, "1.79.2")),
                # corner cases
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 23)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 20)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 23)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 23)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 26)),
                # between
                ((HOST_COMPILER, GCC, 9), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 9), (CXX_STANDARD, 19)),
                ((HOST_COMPILER, GCC, 9), (CXX_STANDARD, 20)),
                # iterate C++ version
                ((CXX_STANDARD, 17), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 18), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 19), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 20), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 21), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 22), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 23), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 24), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 25), (DEVICE_COMPILER, GCC, 11)),
                # older than minumum specified GCC version
                ((HOST_COMPILER, GCC, 6), (CXX_STANDARD, 0)),
                ((HOST_COMPILER, GCC, 6), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, GCC, 6), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 6), (CXX_STANDARD, 20)),
                # newer than maximum specified GCC version
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 23)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 9999)),
            ]
        )

        expected_results: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                # not affected
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((DEVICE_COMPILER, GCC, 6), (BOOST, "1.81.1")),
                ((UBUNTU, "20.04"), (CXX_STANDARD, 17)),
                ((CXX_STANDARD, 17), (BOOST, "1.79.2")),
                # corner cases
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 23)),
                # between
                ((HOST_COMPILER, GCC, 9), (CXX_STANDARD, 17)),
                # iterate C++ version
                ((CXX_STANDARD, 17), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 18), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 19), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 20), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 21), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 22), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 23), (DEVICE_COMPILER, GCC, 11)),
                # older than minumum specified GCC version
                ((HOST_COMPILER, GCC, 6), (CXX_STANDARD, 0)),
                ((HOST_COMPILER, GCC, 6), (CXX_STANDARD, 14)),
                # newer than maximum specified GCC version
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 23)),
            ]
        )

        default_remove_test(
            _remove_unsupported_cxx_versions_for_gcc,
            test_param_value_pairs,
            expected_results,
            self,
        )
        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_unsupported_cxx_versions_for_clang(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                # corner cases
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 23)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 20)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 23)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 23)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 26)),
                # between
                ((HOST_COMPILER, CLANG, 12), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 12), (CXX_STANDARD, 19)),
                ((HOST_COMPILER, CLANG, 13), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 13), (CXX_STANDARD, 20)),
                # older than minumum specified GCC version
                ((HOST_COMPILER, CLANG, 6), (CXX_STANDARD, 0)),
                ((HOST_COMPILER, CLANG, 6), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG, 6), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 6), (CXX_STANDARD, 20)),
                # newer than maximum specified GCC version
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 23)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 9999)),
            ]
        )

        expected_results: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                # corner cases
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 23)),
                # between
                ((HOST_COMPILER, CLANG, 12), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 13), (CXX_STANDARD, 17)),
                # older than minumum specified GCC version
                ((HOST_COMPILER, CLANG, 6), (CXX_STANDARD, 0)),
                ((HOST_COMPILER, CLANG, 6), (CXX_STANDARD, 14)),
                # newer than maximum specified GCC version
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 23)),
            ]
        )

        default_remove_test(
            _remove_unsupported_cxx_versions_for_clang,
            test_param_value_pairs,
            expected_results,
            self,
        )
        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )
