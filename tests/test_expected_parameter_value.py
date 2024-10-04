# pylint: disable=missing-docstring
import unittest
from collections import OrderedDict as OD
import copy

from utils_test import parse_expected_val_pairs, create_diff_parameter_value_pairs
from bashi.types import (
    ParameterValuePair,
)
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.utils import remove_parameter_value_pairs


class TestRemoveExpectedParameterValuePairs(unittest.TestCase):
    def test_remove_parameter_value_pair(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )
        original_length = len(test_param_value_pairs)

        t1_no_remove = copy.deepcopy(test_param_value_pairs)
        t1_expected = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )
        t1_unexpected: List[ParameterValuePair] = sorted(list(set(t1_no_remove) - set(t1_expected)))

        t1_unexpected_test_param_value_pairs: List[ParameterValuePair] = []
        self.assertFalse(
            remove_parameter_value_pairs(
                t1_no_remove,
                t1_unexpected_test_param_value_pairs,
                HOST_COMPILER,
                GCC,
                9,
                DEVICE_COMPILER,
                NVCC,
                11.2,
            )
        )

        t1_no_remove.sort()
        t1_unexpected_test_param_value_pairs.sort()

        self.assertEqual(
            t1_no_remove, t1_expected, create_diff_parameter_value_pairs(t1_no_remove, t1_expected)
        )
        self.assertEqual(len(t1_no_remove), original_length)
        self.assertEqual(
            t1_unexpected_test_param_value_pairs,
            t1_unexpected,
            create_diff_parameter_value_pairs(t1_unexpected_test_param_value_pairs, t1_unexpected),
        )

        t2_remove_single_entry = copy.deepcopy(t1_no_remove)
        t2_expected = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )
        t2_unexpected: List[ParameterValuePair] = sorted(
            list(set(t2_remove_single_entry) - set(t2_expected))
        )

        t2_unexpected_test_param_value_pairs: List[ParameterValuePair] = []
        self.assertTrue(
            remove_parameter_value_pairs(
                t2_remove_single_entry,
                t2_unexpected_test_param_value_pairs,
                HOST_COMPILER,
                GCC,
                10,
                DEVICE_COMPILER,
                NVCC,
                12.0,
            )
        )

        t2_remove_single_entry.sort()
        t2_unexpected_test_param_value_pairs.sort()

        self.assertEqual(
            t2_remove_single_entry,
            t2_expected,
            create_diff_parameter_value_pairs(t2_remove_single_entry, t2_expected),
        )
        self.assertEqual(len(t2_remove_single_entry), original_length - 1)
        self.assertEqual(
            t2_unexpected_test_param_value_pairs,
            t2_unexpected,
            create_diff_parameter_value_pairs(t2_unexpected_test_param_value_pairs, t2_unexpected),
        )

        t3_remove_another_entry = copy.deepcopy(t2_remove_single_entry)
        t3_expected = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 12.0)}),
                ]
            )
        )
        t3_unexpected: List[ParameterValuePair] = sorted(
            list(set(t3_remove_another_entry) - set(t3_expected))
        )

        t3_unexpected_test_param_value_pairs: List[ParameterValuePair] = []
        self.assertTrue(
            remove_parameter_value_pairs(
                t3_remove_another_entry,
                t3_unexpected_test_param_value_pairs,
                CMAKE,
                CMAKE,
                3.23,
                BOOST,
                BOOST,
                1.83,
            )
        )

        t3_remove_another_entry.sort()
        t3_unexpected_test_param_value_pairs.sort()

        self.assertEqual(
            t3_remove_another_entry,
            t3_expected,
            create_diff_parameter_value_pairs(t3_remove_another_entry, t3_expected),
        )
        self.assertEqual(len(t3_remove_another_entry), original_length - 2)
        self.assertEqual(
            t3_unexpected_test_param_value_pairs,
            t3_unexpected,
            create_diff_parameter_value_pairs(t3_unexpected_test_param_value_pairs, t3_unexpected),
        )

    def test_all_white_card(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )
        len_before = len(test_param_value_pairs)

        unexpected_test_param_value_pairs: List[ParameterValuePair] = []
        self.assertTrue(
            remove_parameter_value_pairs(
                test_param_value_pairs,
                unexpected_test_param_value_pairs,
                parameter1=ANY_PARAM,
                value_name1=ANY_NAME,
                value_version1=ANY_VERSION,
                parameter2=ANY_PARAM,
                value_name2=ANY_NAME,
                value_version2=ANY_VERSION,
            )
        )

        self.assertEqual(len(test_param_value_pairs), 0)
        self.assertEqual(len(unexpected_test_param_value_pairs), len_before)

    def test_single_white_card(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (CLANG, 17)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (GCC, 17)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                OD({HOST_COMPILER: (CLANG, 10), BOOST: (BOOST, 1.83)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
            ]
        )
        test_original_len = len(test_param_value_pairs)

        t1_any_version1_param_value_pairs = copy.deepcopy(test_param_value_pairs)
        t1_expected = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (CLANG, 17)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (GCC, 17)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD({HOST_COMPILER: (CLANG, 10), BOOST: (BOOST, 1.83)}),
                    OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                ]
            )
        )
        t1_unexpected: List[ParameterValuePair] = sorted(
            list(set(t1_any_version1_param_value_pairs) - set(t1_expected))
        )

        t1_unexpected_test_param_value_pairs: List[ParameterValuePair] = []
        self.assertTrue(
            remove_parameter_value_pairs(
                t1_any_version1_param_value_pairs,
                t1_unexpected_test_param_value_pairs,
                parameter1=HOST_COMPILER,
                value_name1=GCC,
                value_version1=ANY_VERSION,
                parameter2=DEVICE_COMPILER,
                value_name2=NVCC,
                value_version2=12.0,
            )
        )
        t1_any_version1_param_value_pairs.sort()
        t1_unexpected_test_param_value_pairs.sort()

        self.assertEqual(
            t1_any_version1_param_value_pairs,
            t1_expected,
            create_diff_parameter_value_pairs(t1_any_version1_param_value_pairs, t1_expected),
        )
        self.assertEqual(len(t1_any_version1_param_value_pairs), test_original_len - 2)
        self.assertEqual(
            t1_unexpected_test_param_value_pairs,
            t1_unexpected,
            create_diff_parameter_value_pairs(t1_unexpected_test_param_value_pairs, t1_unexpected),
        )

        t2_any_name1_param_value_pairs = copy.deepcopy(test_param_value_pairs)
        t2_expected = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (CLANG, 17)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (GCC, 17)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD({HOST_COMPILER: (CLANG, 10), BOOST: (BOOST, 1.83)}),
                ]
            )
        )
        t2_unexpected: List[ParameterValuePair] = sorted(
            list(set(t2_any_name1_param_value_pairs) - set(t2_expected))
        )

        t2_unexpected_test_param_value_pairs: List[ParameterValuePair] = []
        self.assertTrue(
            remove_parameter_value_pairs(
                t2_any_name1_param_value_pairs,
                t2_unexpected_test_param_value_pairs,
                parameter1=HOST_COMPILER,
                value_name1=ANY_NAME,
                value_version1=10,
                parameter2=DEVICE_COMPILER,
                value_name2=NVCC,
                value_version2=12.0,
            )
        )

        t2_any_name1_param_value_pairs.sort()
        t2_unexpected_test_param_value_pairs.sort()

        self.assertEqual(
            t2_any_name1_param_value_pairs,
            t2_expected,
            create_diff_parameter_value_pairs(t2_any_name1_param_value_pairs, t2_expected),
        )
        self.assertEqual(len(t2_any_name1_param_value_pairs), test_original_len - 2)
        self.assertEqual(
            t2_unexpected_test_param_value_pairs,
            t2_unexpected,
            create_diff_parameter_value_pairs(t2_unexpected_test_param_value_pairs, t2_unexpected),
        )

    def test_white_card_multi_parameter(self):
        t1_any_parameter_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({BOOST: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (CLANG, 17)}),
                OD({HOST_COMPILER: (CLANG, 17), DEVICE_COMPILER: (CLANG, 16)}),
                OD({CMAKE: (GCC, 10), UBUNTU: (NVCC, 11.2)}),
            ]
        )
        test_original_len = len(t1_any_parameter_param_value_pairs)

        t1_expected = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (CLANG, 17)}),
                    OD({HOST_COMPILER: (CLANG, 17), DEVICE_COMPILER: (CLANG, 16)}),
                ]
            )
        )
        t1_unexpected: List[ParameterValuePair] = sorted(
            list(set(t1_any_parameter_param_value_pairs) - set(t1_expected))
        )

        t1_unexpected_test_param_value_pairs: List[ParameterValuePair] = []
        self.assertTrue(
            remove_parameter_value_pairs(
                t1_any_parameter_param_value_pairs,
                t1_unexpected_test_param_value_pairs,
                parameter1=ANY_PARAM,
                value_name1=GCC,
                value_version1=10,
                parameter2=ANY_PARAM,
                value_name2=NVCC,
                value_version2=11.2,
            )
        )

        t1_any_parameter_param_value_pairs.sort()
        t1_unexpected_test_param_value_pairs.sort()

        self.assertEqual(
            t1_any_parameter_param_value_pairs,
            t1_expected,
            create_diff_parameter_value_pairs(t1_any_parameter_param_value_pairs, t1_expected),
        )
        self.assertEqual(len(t1_any_parameter_param_value_pairs), test_original_len - 3)
        self.assertEqual(
            t1_unexpected_test_param_value_pairs,
            t1_unexpected,
            create_diff_parameter_value_pairs(t1_unexpected_test_param_value_pairs, t1_unexpected),
        )

    def test_remove_all_gcc_host(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (CLANG, 17)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (GCC, 17)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                OD({HOST_COMPILER: (CLANG, 10), BOOST: (BOOST, 1.83)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
            ]
        )
        test_original_len = len(test_param_value_pairs)
        t_expected = sorted(
            parse_expected_val_pairs(
                [
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD({HOST_COMPILER: (CLANG, 10), BOOST: (BOOST, 1.83)}),
                    OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                ]
            )
        )
        t1_unexpected: List[ParameterValuePair] = sorted(
            list(set(test_param_value_pairs) - set(t_expected))
        )

        unexpected_test_param_value_pairs: List[ParameterValuePair] = []
        self.assertTrue(
            remove_parameter_value_pairs(
                test_param_value_pairs,
                unexpected_test_param_value_pairs,
                parameter1=HOST_COMPILER,
                value_name1=GCC,
                value_version1=ANY_VERSION,
            )
        )

        test_param_value_pairs.sort()
        unexpected_test_param_value_pairs.sort()

        self.assertEqual(
            test_param_value_pairs,
            t_expected,
            create_diff_parameter_value_pairs(test_param_value_pairs, t_expected),
        )
        self.assertEqual(len(test_param_value_pairs), test_original_len - 5)
        self.assertEqual(
            unexpected_test_param_value_pairs,
            t1_unexpected,
            create_diff_parameter_value_pairs(unexpected_test_param_value_pairs, t1_unexpected),
        )

    def test_symmetric(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                OD({DEVICE_COMPILER: (NVCC, 11.2), HOST_COMPILER: (GCC, 10)}),
                OD({DEVICE_COMPILER: (NVCC, 12.0), HOST_COMPILER: (GCC, 10)}),
                OD({BOOST: (BOOST, 1.83), CMAKE: (CMAKE, 3.23)}),
            ]
        )
        test_original_len = len(test_param_value_pairs)
        t1_expected = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD({DEVICE_COMPILER: (NVCC, 11.2), HOST_COMPILER: (GCC, 10)}),
                    OD({BOOST: (BOOST, 1.83), CMAKE: (CMAKE, 3.23)}),
                ]
            )
        )
        t1_single_hit_symmetric_param_value_pairs = copy.deepcopy(test_param_value_pairs)
        t1_unexpected: List[ParameterValuePair] = sorted(
            list(set(t1_single_hit_symmetric_param_value_pairs) - set(t1_expected))
        )

        t1_unexpected_test_param_value_pairs: List[ParameterValuePair] = []
        self.assertTrue(
            remove_parameter_value_pairs(
                t1_single_hit_symmetric_param_value_pairs,
                t1_unexpected_test_param_value_pairs,
                parameter1=HOST_COMPILER,
                value_name1=GCC,
                value_version1=10,
                parameter2=DEVICE_COMPILER,
                value_name2=NVCC,
                value_version2=12.0,
            )
        )

        t1_single_hit_symmetric_param_value_pairs.sort()
        t1_unexpected_test_param_value_pairs.sort()

        self.assertEqual(
            t1_single_hit_symmetric_param_value_pairs,
            t1_expected,
            create_diff_parameter_value_pairs(
                t1_single_hit_symmetric_param_value_pairs, t1_expected
            ),
        )
        self.assertEqual(len(t1_single_hit_symmetric_param_value_pairs), test_original_len - 2)
        self.assertEqual(
            t1_unexpected_test_param_value_pairs,
            t1_unexpected,
            create_diff_parameter_value_pairs(t1_unexpected_test_param_value_pairs, t1_unexpected),
        )

        t2_single_hit_no_symmetric_param_value_pairs = copy.deepcopy(test_param_value_pairs)
        t2_expected = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD({DEVICE_COMPILER: (NVCC, 11.2), HOST_COMPILER: (GCC, 10)}),
                    OD({DEVICE_COMPILER: (NVCC, 12.0), HOST_COMPILER: (GCC, 10)}),
                    OD({BOOST: (BOOST, 1.83), CMAKE: (CMAKE, 3.23)}),
                ]
            )
        )
        t2_unexpected: List[ParameterValuePair] = sorted(
            list(set(t2_single_hit_no_symmetric_param_value_pairs) - set(t2_expected))
        )

        t2_unexpected_test_param_value_pairs: List[ParameterValuePair] = []
        self.assertTrue(
            remove_parameter_value_pairs(
                t2_single_hit_no_symmetric_param_value_pairs,
                t2_unexpected_test_param_value_pairs,
                parameter1=HOST_COMPILER,
                value_name1=GCC,
                value_version1=10,
                parameter2=DEVICE_COMPILER,
                value_name2=NVCC,
                value_version2=12.0,
                symmetric=False,
            )
        )

        t2_single_hit_no_symmetric_param_value_pairs.sort()
        t2_unexpected_test_param_value_pairs.sort()

        self.assertEqual(
            t2_single_hit_no_symmetric_param_value_pairs,
            t2_expected,
            create_diff_parameter_value_pairs(
                t2_single_hit_no_symmetric_param_value_pairs, t2_expected
            ),
        )
        self.assertEqual(len(t2_single_hit_no_symmetric_param_value_pairs), test_original_len - 1)
        self.assertEqual(
            t2_unexpected_test_param_value_pairs,
            t2_unexpected,
            create_diff_parameter_value_pairs(t2_unexpected_test_param_value_pairs, t2_unexpected),
        )

        t3_single_hit_no_symmetric_param_value_pairs = copy.deepcopy(test_param_value_pairs)
        t3_expected = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD({DEVICE_COMPILER: (NVCC, 11.2), HOST_COMPILER: (GCC, 10)}),
                    OD({BOOST: (BOOST, 1.83), CMAKE: (CMAKE, 3.23)}),
                ]
            )
        )
        t3_unexpected: List[ParameterValuePair] = sorted(
            list(set(t3_single_hit_no_symmetric_param_value_pairs) - set(t3_expected))
        )

        t3_unexpected_test_param_value_pairs: List[ParameterValuePair] = []
        self.assertTrue(
            remove_parameter_value_pairs(
                t3_single_hit_no_symmetric_param_value_pairs,
                t3_unexpected_test_param_value_pairs,
                parameter1=DEVICE_COMPILER,
                value_name1=NVCC,
                value_version1=12.0,
                parameter2=HOST_COMPILER,
                value_name2=GCC,
                value_version2=10,
                symmetric=False,
            )
        )

        t3_single_hit_no_symmetric_param_value_pairs.sort()
        t3_unexpected_test_param_value_pairs.sort()

        self.assertEqual(
            t3_single_hit_no_symmetric_param_value_pairs,
            t3_expected,
            create_diff_parameter_value_pairs(
                t3_single_hit_no_symmetric_param_value_pairs, t3_expected
            ),
        )
        self.assertEqual(len(t3_single_hit_no_symmetric_param_value_pairs), test_original_len - 1)
        self.assertEqual(
            t3_unexpected_test_param_value_pairs,
            t3_unexpected,
            create_diff_parameter_value_pairs(t3_unexpected_test_param_value_pairs, t3_unexpected),
        )

        t4_multi_hit_symmetric_param_value_pairs = copy.deepcopy(test_param_value_pairs)
        t4_expected = sorted(
            parse_expected_val_pairs(
                [
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD({BOOST: (BOOST, 1.83), CMAKE: (CMAKE, 3.23)}),
                ]
            )
        )
        t4_unexpected: List[ParameterValuePair] = sorted(
            list(set(t4_multi_hit_symmetric_param_value_pairs) - set(t4_expected))
        )

        t4_unexpected_test_param_value_pairs: List[ParameterValuePair] = []
        self.assertTrue(
            remove_parameter_value_pairs(
                t4_multi_hit_symmetric_param_value_pairs,
                t4_unexpected_test_param_value_pairs,
                parameter1=HOST_COMPILER,
                value_name1=GCC,
                value_version1=ANY_VERSION,
            )
        )

        t4_multi_hit_symmetric_param_value_pairs.sort()
        t4_unexpected_test_param_value_pairs.sort()

        self.assertEqual(
            t4_multi_hit_symmetric_param_value_pairs,
            t4_expected,
            create_diff_parameter_value_pairs(
                t4_multi_hit_symmetric_param_value_pairs, t4_expected
            ),
        )
        self.assertEqual(len(t4_multi_hit_symmetric_param_value_pairs), test_original_len - 4)
        self.assertEqual(
            t4_unexpected_test_param_value_pairs,
            t4_unexpected,
            create_diff_parameter_value_pairs(t4_unexpected_test_param_value_pairs, t4_unexpected),
        )

        t5_multi_hit_no_symmetric_param_value_pairs = copy.deepcopy(test_param_value_pairs)
        t5_expected = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD({BOOST: (BOOST, 1.83), CMAKE: (CMAKE, 3.23)}),
                ]
            )
        )
        t5_unexpected: List[ParameterValuePair] = sorted(
            list(set(t5_multi_hit_no_symmetric_param_value_pairs) - set(t5_expected))
        )

        t5_unexpected_test_param_value_pairs: List[ParameterValuePair] = []

        self.assertTrue(
            remove_parameter_value_pairs(
                t5_multi_hit_no_symmetric_param_value_pairs,
                t5_unexpected_test_param_value_pairs,
                parameter2=HOST_COMPILER,
                value_name2=GCC,
                value_version2=ANY_VERSION,
                symmetric=False,
            )
        )

        t5_multi_hit_no_symmetric_param_value_pairs.sort()
        t5_unexpected_test_param_value_pairs.sort()

        self.assertEqual(
            t5_multi_hit_no_symmetric_param_value_pairs,
            t5_expected,
            create_diff_parameter_value_pairs(
                t5_multi_hit_no_symmetric_param_value_pairs, t5_expected
            ),
        )
        self.assertEqual(len(t5_multi_hit_no_symmetric_param_value_pairs), test_original_len - 2)
        self.assertEqual(
            t5_unexpected_test_param_value_pairs,
            t5_unexpected,
            create_diff_parameter_value_pairs(t5_unexpected_test_param_value_pairs, t5_unexpected),
        )
