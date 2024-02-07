# pylint: disable=missing-docstring
import unittest
from typing import List, Dict
from collections import OrderedDict
import io
import packaging.version as pkv

# allpairspy has no type hints
from allpairspy import AllPairs  # type: ignore
from utils_test import parse_param_val, parse_param_vals, parse_expected_val_pairs
from bashi.types import (
    Parameter,
    ParameterValue,
    ParameterValueSingle,
    ParameterValuePair,
    ParameterValueMatrix,
    CombinationList,
)
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.utils import (
    get_expected_parameter_value_pairs,
    check_parameter_value_pair_in_combination_list,
    remove_parameter_value_pair,
    create_parameter_value_pair,
)


class TestCreateParameterValuePair(unittest.TestCase):
    def test_create_parameter_value_pair_type_str(self):
        param1 = "param1"
        param2 = "param2"
        name1 = "name1"
        name2 = "name2"
        ver1 = "1.0"
        ver2 = "2.0"

        created_param_val_pair = create_parameter_value_pair(
            param1, name1, ver1, param2, name2, ver2
        )

        expected_param_val_pair = ParameterValuePair(
            ParameterValueSingle(
                param1,
                ParameterValue(name1, pkv.parse(ver1)),
            ),
            ParameterValueSingle(
                param2,
                ParameterValue(name2, pkv.parse(ver2)),
            ),
        )

        self.assertEqual(type(created_param_val_pair), type(expected_param_val_pair))

        self.assertEqual(
            created_param_val_pair,
            expected_param_val_pair,
        )

    def test_create_parameter_value_pair_type_float(self):
        param1 = "param1"
        param2 = "param2"
        name1 = "name1"
        name2 = "name2"
        ver1 = 1.0
        ver2 = 2.0

        created_param_val_pair = create_parameter_value_pair(
            param1, name1, ver1, param2, name2, ver2
        )

        expected_param_val_pair = ParameterValuePair(
            ParameterValueSingle(
                param1,
                ParameterValue(name1, pkv.parse(str(ver1))),
            ),
            ParameterValueSingle(
                param2,
                ParameterValue(name2, pkv.parse(str(ver2))),
            ),
        )

        self.assertEqual(type(created_param_val_pair), type(expected_param_val_pair))

        self.assertEqual(
            created_param_val_pair,
            expected_param_val_pair,
        )

    def test_create_parameter_value_pair_type_int(self):
        param1 = "param1"
        param2 = "param2"
        name1 = "name1"
        name2 = "name2"
        ver1 = 1
        ver2 = 2

        created_param_val_pair = create_parameter_value_pair(
            param1, name1, ver1, param2, name2, ver2
        )

        expected_param_val_pair = ParameterValuePair(
            ParameterValueSingle(
                param1,
                ParameterValue(name1, pkv.parse(str(ver1))),
            ),
            ParameterValueSingle(
                param2,
                ParameterValue(name2, pkv.parse(str(ver2))),
            ),
        )

        self.assertEqual(type(created_param_val_pair), type(expected_param_val_pair))

        self.assertEqual(
            created_param_val_pair,
            expected_param_val_pair,
        )

    def test_create_parameter_value_pair_type_version(self):
        param1 = "param1"
        param2 = "param2"
        name1 = "name1"
        name2 = "name2"
        ver1 = pkv.parse("1.0")
        ver2 = pkv.parse("2.0")

        created_param_val_pair = create_parameter_value_pair(
            param1, name1, ver1, param2, name2, ver2
        )

        expected_param_val_pair = ParameterValuePair(
            ParameterValueSingle(
                param1,
                ParameterValue(name1, ver1),
            ),
            ParameterValueSingle(
                param2,
                ParameterValue(name2, ver2),
            ),
        )

        self.assertEqual(type(created_param_val_pair), type(expected_param_val_pair))

        self.assertEqual(
            created_param_val_pair,
            expected_param_val_pair,
        )

    def test_create_parameter_value_pair_type_mixed(self):
        param1 = "param1"
        param2 = "param2"
        name1 = "name1"
        name2 = "name2"
        ver1 = pkv.parse("1.0")
        ver2 = "2.0"

        created_param_val_pair = create_parameter_value_pair(
            param1, name1, ver1, param2, name2, ver2
        )

        expected_param_val_pair = ParameterValuePair(
            ParameterValueSingle(
                param1,
                ParameterValue(name1, ver1),
            ),
            ParameterValueSingle(
                param2,
                ParameterValue(name2, pkv.parse(ver2)),
            ),
        )

        self.assertEqual(type(created_param_val_pair), type(expected_param_val_pair))

        self.assertEqual(
            created_param_val_pair,
            expected_param_val_pair,
        )


class TestExpectedValuePairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.param_matrix: ParameterValueMatrix = OrderedDict()

        cls.param_matrix[HOST_COMPILER] = parse_param_vals(
            [(GCC, 10), (GCC, 11), (GCC, 12), (CLANG, 16), (CLANG, 17)]
        )
        cls.param_matrix[DEVICE_COMPILER] = parse_param_vals(
            [(NVCC, 11.2), (NVCC, 12.0), (GCC, 10), (GCC, 11)]
        )
        cls.param_matrix[CMAKE] = parse_param_vals([(CMAKE, 3.22), (CMAKE, 3.23)])
        cls.param_matrix[BOOST] = parse_param_vals([(BOOST, 1.81), (BOOST, 1.82), (BOOST, 1.83)])

        cls.generated_parameter_value_pairs: List[ParameterValuePair] = (
            get_expected_parameter_value_pairs(cls.param_matrix)
        )

        OD = OrderedDict

        cls.expected_param_val_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (GCC, 10)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (GCC, 11)}),
                OD({HOST_COMPILER: (GCC, 10), CMAKE: (CMAKE, 3.22)}),
                OD({HOST_COMPILER: (GCC, 10), CMAKE: (CMAKE, 3.23)}),
                OD({HOST_COMPILER: (GCC, 10), BOOST: (BOOST, 1.81)}),
                OD({HOST_COMPILER: (GCC, 10), BOOST: (BOOST, 1.82)}),
                OD({HOST_COMPILER: (GCC, 10), BOOST: (BOOST, 1.83)}),
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (GCC, 10)}),
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (GCC, 11)}),
                OD({HOST_COMPILER: (GCC, 11), CMAKE: (CMAKE, 3.22)}),
                OD({HOST_COMPILER: (GCC, 11), CMAKE: (CMAKE, 3.23)}),
                OD({HOST_COMPILER: (GCC, 11), BOOST: (BOOST, 1.81)}),
                OD({HOST_COMPILER: (GCC, 11), BOOST: (BOOST, 1.82)}),
                OD({HOST_COMPILER: (GCC, 11), BOOST: (BOOST, 1.83)}),
                OD({HOST_COMPILER: (GCC, 12), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 12), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 12), DEVICE_COMPILER: (GCC, 10)}),
                OD({HOST_COMPILER: (GCC, 12), DEVICE_COMPILER: (GCC, 11)}),
                OD({HOST_COMPILER: (GCC, 12), CMAKE: (CMAKE, 3.22)}),
                OD({HOST_COMPILER: (GCC, 12), CMAKE: (CMAKE, 3.23)}),
                OD({HOST_COMPILER: (GCC, 12), BOOST: (BOOST, 1.81)}),
                OD({HOST_COMPILER: (GCC, 12), BOOST: (BOOST, 1.82)}),
                OD({HOST_COMPILER: (GCC, 12), BOOST: (BOOST, 1.83)}),
                OD({HOST_COMPILER: (CLANG, 16), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (CLANG, 16), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (CLANG, 16), DEVICE_COMPILER: (GCC, 10)}),
                OD({HOST_COMPILER: (CLANG, 16), DEVICE_COMPILER: (GCC, 11)}),
                OD({HOST_COMPILER: (CLANG, 16), CMAKE: (CMAKE, 3.22)}),
                OD({HOST_COMPILER: (CLANG, 16), CMAKE: (CMAKE, 3.23)}),
                OD({HOST_COMPILER: (CLANG, 16), BOOST: (BOOST, 1.81)}),
                OD({HOST_COMPILER: (CLANG, 16), BOOST: (BOOST, 1.82)}),
                OD({HOST_COMPILER: (CLANG, 16), BOOST: (BOOST, 1.83)}),
                OD({HOST_COMPILER: (CLANG, 17), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (CLANG, 17), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (CLANG, 17), DEVICE_COMPILER: (GCC, 10)}),
                OD({HOST_COMPILER: (CLANG, 17), DEVICE_COMPILER: (GCC, 11)}),
                OD({HOST_COMPILER: (CLANG, 17), CMAKE: (CMAKE, 3.22)}),
                OD({HOST_COMPILER: (CLANG, 17), CMAKE: (CMAKE, 3.23)}),
                OD({HOST_COMPILER: (CLANG, 17), BOOST: (BOOST, 1.81)}),
                OD({HOST_COMPILER: (CLANG, 17), BOOST: (BOOST, 1.82)}),
                OD({HOST_COMPILER: (CLANG, 17), BOOST: (BOOST, 1.83)}),
                OD({DEVICE_COMPILER: (NVCC, 11.2), CMAKE: (CMAKE, 3.22)}),
                OD({DEVICE_COMPILER: (NVCC, 11.2), CMAKE: (CMAKE, 3.23)}),
                OD({DEVICE_COMPILER: (NVCC, 11.2), BOOST: (BOOST, 1.81)}),
                OD({DEVICE_COMPILER: (NVCC, 11.2), BOOST: (BOOST, 1.82)}),
                OD({DEVICE_COMPILER: (NVCC, 11.2), BOOST: (BOOST, 1.83)}),
                OD({DEVICE_COMPILER: (NVCC, 12.0), CMAKE: (CMAKE, 3.22)}),
                OD({DEVICE_COMPILER: (NVCC, 12.0), CMAKE: (CMAKE, 3.23)}),
                OD({DEVICE_COMPILER: (NVCC, 12.0), BOOST: (BOOST, 1.81)}),
                OD({DEVICE_COMPILER: (NVCC, 12.0), BOOST: (BOOST, 1.82)}),
                OD({DEVICE_COMPILER: (NVCC, 12.0), BOOST: (BOOST, 1.83)}),
                OD({DEVICE_COMPILER: (GCC, 10), CMAKE: (CMAKE, 3.22)}),
                OD({DEVICE_COMPILER: (GCC, 10), CMAKE: (CMAKE, 3.23)}),
                OD({DEVICE_COMPILER: (GCC, 10), BOOST: (BOOST, 1.81)}),
                OD({DEVICE_COMPILER: (GCC, 10), BOOST: (BOOST, 1.82)}),
                OD({DEVICE_COMPILER: (GCC, 10), BOOST: (BOOST, 1.83)}),
                OD({DEVICE_COMPILER: (GCC, 11), CMAKE: (CMAKE, 3.22)}),
                OD({DEVICE_COMPILER: (GCC, 11), CMAKE: (CMAKE, 3.23)}),
                OD({DEVICE_COMPILER: (GCC, 11), BOOST: (BOOST, 1.81)}),
                OD({DEVICE_COMPILER: (GCC, 11), BOOST: (BOOST, 1.82)}),
                OD({DEVICE_COMPILER: (GCC, 11), BOOST: (BOOST, 1.83)}),
                OD({CMAKE: (CMAKE, 3.22), BOOST: (BOOST, 1.81)}),
                OD({CMAKE: (CMAKE, 3.22), BOOST: (BOOST, 1.82)}),
                OD({CMAKE: (CMAKE, 3.22), BOOST: (BOOST, 1.83)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.81)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.82)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        ppv = parse_param_val
        cls.handwritten_comb_list: CombinationList = [
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 10)),
                    DEVICE_COMPILER: ppv((NVCC, 11.2)),
                    CMAKE: ppv((CMAKE, 3.22)),
                    BOOST: ppv((BOOST, 1.81)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 10)),
                    DEVICE_COMPILER: ppv((NVCC, 12.0)),
                    CMAKE: ppv((CMAKE, 3.22)),
                    BOOST: ppv((BOOST, 1.82)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((CLANG, 16)),
                    DEVICE_COMPILER: ppv((CLANG, 16)),
                    CMAKE: ppv((CMAKE, 3.23)),
                    BOOST: ppv((BOOST, 1.83)),
                }
            ),
        ]

        cls.handwritten_all_existing_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (CLANG, 16), DEVICE_COMPILER: (CLANG, 16)}),
                OD({HOST_COMPILER: (GCC, 10), CMAKE: (CMAKE, 3.22)}),
                OD({HOST_COMPILER: (CLANG, 16), CMAKE: (CMAKE, 3.23)}),
                OD({HOST_COMPILER: (GCC, 10), BOOST: (BOOST, 1.81)}),
                OD({HOST_COMPILER: (GCC, 10), BOOST: (BOOST, 1.82)}),
                OD({HOST_COMPILER: (CLANG, 16), BOOST: (BOOST, 1.83)}),
                OD({DEVICE_COMPILER: (NVCC, 11.2), CMAKE: (CMAKE, 3.22)}),
                OD({DEVICE_COMPILER: (NVCC, 12.0), CMAKE: (CMAKE, 3.22)}),
                OD({DEVICE_COMPILER: (CLANG, 16), CMAKE: (CMAKE, 3.23)}),
                OD({DEVICE_COMPILER: (NVCC, 11.2), BOOST: (BOOST, 1.81)}),
                OD({DEVICE_COMPILER: (NVCC, 12.0), BOOST: (BOOST, 1.82)}),
                OD({DEVICE_COMPILER: (CLANG, 16), BOOST: (BOOST, 1.83)}),
                OD({CMAKE: (CMAKE, 3.22), BOOST: (BOOST, 1.81)}),
                OD({CMAKE: (CMAKE, 3.22), BOOST: (BOOST, 1.82)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

    def test_expected_value_pairs(self):
        self.assertTrue(
            len(self.expected_param_val_pairs) == len(self.generated_parameter_value_pairs),
            f"\nlen(expected_param_val_pairs): {len(self.expected_param_val_pairs)}\n"
            f"len(generated_parameter_value_pairs): {len(self.generated_parameter_value_pairs)}",
        )

        missing_expected_param = False
        for ex_param_val_pair in self.expected_param_val_pairs:
            try:
                self.assertTrue(ex_param_val_pair in self.generated_parameter_value_pairs)
            except AssertionError:
                missing_expected_param = True
                print(
                    f"{ex_param_val_pair} was not found in the generated parameter-value-pair list"
                )

        self.assertFalse(missing_expected_param)

        missing_generated_param = False
        for gen_param_val_pair in self.generated_parameter_value_pairs:
            try:
                self.assertTrue(gen_param_val_pair in self.expected_param_val_pairs)
            except AssertionError:
                missing_generated_param = True
                print(
                    f"{gen_param_val_pair} was not found in the expected parameter-value-pair list"
                )

        self.assertFalse(missing_generated_param)

    def test_check_parameter_value_pair_in_combination_list_empty_input(self):
        self.assertTrue(
            check_parameter_value_pair_in_combination_list(self.handwritten_comb_list, [])
        )

    def test_check_parameter_value_pair_in_combination_list_less_valid_input(self):
        OD = OrderedDict

        # all pairs exists in the combination list, but not all pairs are tested
        self.assertTrue(
            check_parameter_value_pair_in_combination_list(
                self.handwritten_comb_list,
                parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                        OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                        OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    ]
                ),
            )
        )

    def test_check_parameter_value_pair_in_combination_list_complete_valid_input(self):
        # test all existing pairs
        self.assertTrue(
            check_parameter_value_pair_in_combination_list(
                self.handwritten_comb_list, self.handwritten_all_existing_pairs
            )
        )

    def test_check_parameter_value_pair_in_combination_list_single_wrong_input(self):
        OD = OrderedDict

        single_wrong_pair = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.2)}),
            ]
        )

        output_wrong_single_pair = io.StringIO()
        self.assertFalse(
            check_parameter_value_pair_in_combination_list(
                self.handwritten_comb_list,
                single_wrong_pair,
                output_wrong_single_pair,
            )
        )
        output_wrong_single_pair_expected_str = (
            str(single_wrong_pair[0]) + " is missing in combination list"
        )
        self.assertTrue(
            output_wrong_single_pair.getvalue().rstrip() == output_wrong_single_pair_expected_str,
            (
                f"\nGet:      {output_wrong_single_pair.getvalue().rstrip()}\n"
                f"Expected: {output_wrong_single_pair_expected_str}"
            ),
        )

    def test_check_parameter_value_pair_in_combination_list_many_wrong_input(self):
        OD = OrderedDict

        many_wrong_pairs = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({CMAKE: (CMAKE, 2.23), DEVICE_COMPILER: (BOOST, 1.83)}),
                OD({DEVICE_COMPILER: (NVCC, 12.0), BOOST: (BOOST, 1.84)}),
            ]
        )

        expected_output_many_wrong_pairs_list: List[str] = []
        for pair in many_wrong_pairs:
            expected_output_many_wrong_pairs_list.append(
                str(pair) + " is missing in combination list"
            )

        expected_output_many_wrong_pairs_list.sort()

        output_wrong_many_pairs = io.StringIO()
        self.assertFalse(
            check_parameter_value_pair_in_combination_list(
                self.handwritten_comb_list,
                many_wrong_pairs,
                output_wrong_many_pairs,
            )
        )

        output_wrong_many_pairs_list: List[str] = (
            output_wrong_many_pairs.getvalue().rstrip().split("\n")
        )
        output_wrong_many_pairs_list.sort()

        self.assertEqual(len(output_wrong_many_pairs_list), 3)

        self.assertEqual(output_wrong_many_pairs_list, expected_output_many_wrong_pairs_list)

    def test_check_parameter_value_pair_in_combination_list_complete_list_plus_wrong_input(self):
        OD = OrderedDict

        many_wrong_pairs = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({CMAKE: (CMAKE, 2.23), DEVICE_COMPILER: (BOOST, 1.83)}),
                OD({DEVICE_COMPILER: (NVCC, 12.0), BOOST: (BOOST, 1.84)}),
            ]
        )

        all_pairs_plus_wrong_input = self.handwritten_all_existing_pairs + many_wrong_pairs

        expected_output_many_wrong_pairs_list: List[str] = []
        for pair in many_wrong_pairs:
            expected_output_many_wrong_pairs_list.append(
                str(pair) + " is missing in combination list"
            )

        expected_output_many_wrong_pairs_list.sort()

        output_wrong_many_pairs = io.StringIO()
        self.assertFalse(
            check_parameter_value_pair_in_combination_list(
                self.handwritten_comb_list,
                all_pairs_plus_wrong_input,
                output_wrong_many_pairs,
            )
        )

        output_wrong_many_pairs_list: List[str] = (
            output_wrong_many_pairs.getvalue().rstrip().split("\n")
        )
        output_wrong_many_pairs_list.sort()

        self.assertEqual(len(output_wrong_many_pairs_list), 3)

        self.assertEqual(output_wrong_many_pairs_list, expected_output_many_wrong_pairs_list)

    def test_unrestricted_allpairspy_generator(self):
        comb_list: CombinationList = []
        # pylance shows a warning, because it cannot determine the concrete type of a namedtuple,
        # which is returned by AllPairs
        for all_pair in AllPairs(parameters=self.param_matrix):  # type: ignore
            comb: Combination = OrderedDict()
            for index, param in enumerate(all_pair._fields):  # type: ignore
                comb[param] = all_pair[index]  # type: ignore
            comb_list.append(comb)

        self.assertTrue(
            check_parameter_value_pair_in_combination_list(comb_list, self.expected_param_val_pairs)
        )


class TestRemoveExpectedParameterValuePair(unittest.TestCase):
    def test_remove_parameter_value_pair(self):
        OD = OrderedDict

        expected_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )
        original_length = len(expected_param_value_pairs)

        self.assertFalse(
            remove_parameter_value_pair(
                create_parameter_value_pair(HOST_COMPILER, GCC, 9, DEVICE_COMPILER, NVCC, 11.2),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length)

        self.assertTrue(
            remove_parameter_value_pair(
                create_parameter_value_pair(HOST_COMPILER, GCC, 10, DEVICE_COMPILER, NVCC, 12.0),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length - 1)

        self.assertTrue(
            remove_parameter_value_pair(
                create_parameter_value_pair(CMAKE, CMAKE, 3.23, BOOST, BOOST, 1.83),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length - 2)

    def test_remove_parameter_value_single(self):
        OD = OrderedDict
        ppv = parse_param_val

        expected_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )
        original_length = len(expected_param_value_pairs)

        # all_versions=True is not support for ParameterValueSingle
        self.assertRaises(
            RuntimeError,
            remove_parameter_value_pair,
            ParameterValueSingle(HOST_COMPILER, ppv((GCC, 12))),
            expected_param_value_pairs,
            True,
        )

        self.assertFalse(
            remove_parameter_value_pair(
                ParameterValueSingle(HOST_COMPILER, ppv((GCC, 12))),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length)

        self.assertFalse(
            remove_parameter_value_pair(
                ParameterValueSingle(HOST_COMPILER, ppv((CLANG, 12))),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length)

        self.assertFalse(
            remove_parameter_value_pair(
                ParameterValueSingle(UBUNTU, ppv((UBUNTU, 20.04))),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length)

        self.assertTrue(
            remove_parameter_value_pair(
                ParameterValueSingle(HOST_COMPILER, ppv((GCC, 9))),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length - 2)
        self.assertEqual(
            expected_param_value_pairs,
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            ),
        )

    def test_remove_parameter_value_pair_all_versions(self):
        versions = {
            GCC: [9, 10, 11, 12, 13],
            CLANG: [13, 14, 15, 16, 17],
            NVCC: [11.0, 11.1, 11.2, 11.3, 11.4],
            HIPCC: [5.0, 5.1, 5.2, 5.3],
            CMAKE: [3.22, 3.23, 3.24],
            BOOST: [1.80, 1.81, 1.82],
        }

        param_val_matrix: ParameterValueMatrix = OrderedDict()
        for compiler in [HOST_COMPILER, DEVICE_COMPILER]:
            param_val_matrix[compiler] = []
            for compiler_name in [GCC, CLANG, NVCC, HIPCC]:
                for compiler_version in versions[compiler_name]:
                    param_val_matrix[compiler].append(
                        ParameterValue(compiler_name, pkv.parse(str(compiler_version)))
                    )

        for sw in [CMAKE, BOOST]:
            param_val_matrix[sw] = []
            for version in versions[sw]:
                param_val_matrix[sw].append(ParameterValue(sw, pkv.parse(str(version))))

        reduced_param_value_pairs = get_expected_parameter_value_pairs(param_val_matrix)

        expected_number_of_reduced_pairs = len(reduced_param_value_pairs)

        expected_reduced_param_value_pairs = reduced_param_value_pairs.copy()

        # remove single value to verify that default flag is working
        example_single_pair = create_parameter_value_pair(
            HOST_COMPILER,
            NVCC,
            11.0,
            DEVICE_COMPILER,
            NVCC,
            11.3,
        )

        expected_reduced_param_value_pairs.remove(example_single_pair)

        self.assertTrue(
            remove_parameter_value_pair(
                to_remove=example_single_pair,
                parameter_value_pairs=reduced_param_value_pairs,
            )
        )

        # remove single entry
        expected_number_of_reduced_pairs -= 1
        self.assertEqual(len(reduced_param_value_pairs), expected_number_of_reduced_pairs)

        reduced_param_value_pairs.sort()
        expected_reduced_param_value_pairs.sort()
        self.assertEqual(reduced_param_value_pairs, expected_reduced_param_value_pairs)

        # remove all expected tuples, where host and device compiler is nvcc
        def filter_function1(param_val_pair: ParameterValuePair) -> bool:
            if (
                param_val_pair.first.parameter == HOST_COMPILER
                and param_val_pair.second.parameter == DEVICE_COMPILER
            ):
                if (
                    param_val_pair.first.parameterValue.name == NVCC
                    and param_val_pair.second.parameterValue.name == NVCC
                ):
                    return False

            return True

        expected_reduced_param_value_pairs[:] = list(
            filter(filter_function1, expected_reduced_param_value_pairs)
        )

        self.assertTrue(
            remove_parameter_value_pair(
                to_remove=create_parameter_value_pair(
                    HOST_COMPILER,
                    NVCC,
                    0,
                    DEVICE_COMPILER,
                    NVCC,
                    0,
                ),
                parameter_value_pairs=reduced_param_value_pairs,
                all_versions=True,
            )
        )

        # remove number of pairs, where host and device compiler is nvcc
        # -1 because we removed already a combination manually before
        expected_number_of_reduced_pairs -= len(versions[NVCC]) * len(versions[NVCC]) - 1
        self.assertEqual(len(reduced_param_value_pairs), expected_number_of_reduced_pairs)

        reduced_param_value_pairs.sort()
        expected_reduced_param_value_pairs.sort()
        self.assertEqual(reduced_param_value_pairs, expected_reduced_param_value_pairs)

        # remove all combinations where HIPCC is the host compiler and nvcc the device compiler
        def filter_function2(param_val_pair: ParameterValuePair) -> bool:
            if (
                param_val_pair.first.parameter == HOST_COMPILER
                and param_val_pair.second.parameter == DEVICE_COMPILER
            ):
                if (
                    param_val_pair.first.parameterValue.name == HIPCC
                    and param_val_pair.second.parameterValue.name == NVCC
                ):
                    return False

            return True

        expected_reduced_param_value_pairs[:] = list(
            filter(filter_function2, expected_reduced_param_value_pairs)
        )

        self.assertTrue(
            remove_parameter_value_pair(
                to_remove=create_parameter_value_pair(
                    HOST_COMPILER,
                    HIPCC,
                    0,
                    DEVICE_COMPILER,
                    NVCC,
                    0,
                ),
                parameter_value_pairs=reduced_param_value_pairs,
                all_versions=True,
            )
        )

        # remove number pairs, where host compiler is HIPCC and device compiler is nvcc
        expected_number_of_reduced_pairs -= len(versions[HIPCC]) * len(versions[NVCC])
        self.assertEqual(len(reduced_param_value_pairs), expected_number_of_reduced_pairs)

        reduced_param_value_pairs.sort()
        expected_reduced_param_value_pairs.sort()
        self.assertEqual(reduced_param_value_pairs, expected_reduced_param_value_pairs)
