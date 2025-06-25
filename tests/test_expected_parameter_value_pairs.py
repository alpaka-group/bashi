# pylint: disable=missing-docstring
import unittest
from typing import List, Dict
from collections import OrderedDict as OD
import io
import packaging.version as pkv
import random

from utils_test import (
    parse_param_val,
    parse_param_vals,
    parse_expected_val_pairs,
)
from covertable import make
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
    check_unexpected_parameter_value_pair_in_combination_list,
    create_parameter_value_pair,
    get_nice_paremter_value_pair_str,
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
        cls.param_matrix: ParameterValueMatrix = OD()

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
            f"MISSING in combination list: "
            f"{get_nice_paremter_value_pair_str(single_wrong_pair[0])}"
        )
        self.assertTrue(
            output_wrong_single_pair.getvalue().rstrip() == output_wrong_single_pair_expected_str,
            (
                f"\nGet:      {output_wrong_single_pair.getvalue().rstrip()}\n"
                f"Expected: {output_wrong_single_pair_expected_str}"
            ),
        )

    def test_check_parameter_value_pair_in_combination_list_many_wrong_input(self):
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
                f"MISSING in combination list: {get_nice_paremter_value_pair_str(pair)}",
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
                f"MISSING in combination list: {get_nice_paremter_value_pair_str(pair)}",
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

    def test_check_unexpected_parameter_value_pair_in_combination_list_all_empty(self):
        self.assertTrue(
            check_unexpected_parameter_value_pair_in_combination_list(
                [],
                [],
            )
        )

    def test_check_unexpected_parameter_value_pair_in_combination_list_empty_input(self):
        self.assertTrue(
            check_unexpected_parameter_value_pair_in_combination_list(
                [],
                parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (GCC, 10), BOOST: (BOOST, 1.82)}),
                        OD({DEVICE_COMPILER: (CLANG, 16), CMAKE: (CMAKE, 3.23)}),
                        OD({DEVICE_COMPILER: (CLANG, 16), BOOST: (BOOST, 1.83)}),
                        OD({HOST_COMPILER: (GCC, 10), CMAKE: (CMAKE, 3.22)}),
                    ]
                ),
            )
        )

    def test_check_unexpected_parameter_value_pair_in_combination_list_empty_search_list(self):
        self.assertTrue(
            check_unexpected_parameter_value_pair_in_combination_list(
                self.handwritten_comb_list, []
            ),
        )

    def test_check_unexpected_parameter_value_pair_in_combination_list_non_existing_entries(self):
        # non of the pairs are in the combination list
        self.assertTrue(
            check_unexpected_parameter_value_pair_in_combination_list(
                self.handwritten_comb_list,
                parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (NVCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                        OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (CLANG, 7)}),
                        OD({HOST_COMPILER: (NVCC, 12.2), DEVICE_COMPILER: (CMAKE, 3.30)}),
                    ]
                ),
            )
        )

    def test_check_unexpected_parameter_value_pair_in_combination_list_existing_entries(self):
        # all of the pairs are in the combination list
        existing_parameter_value_pairs = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), BOOST: (BOOST, 1.82)}),
                OD({DEVICE_COMPILER: (CLANG, 16), CMAKE: (CMAKE, 3.23)}),
                OD({DEVICE_COMPILER: (CLANG, 16), BOOST: (BOOST, 1.83)}),
                OD({HOST_COMPILER: (GCC, 10), CMAKE: (CMAKE, 3.22)}),
            ]
        )

        error_output = io.StringIO()
        self.assertFalse(
            check_unexpected_parameter_value_pair_in_combination_list(
                self.handwritten_comb_list,
                existing_parameter_value_pairs,
                error_output,
            )
        )

        error_list = error_output.getvalue().rstrip().split("\n")
        self.assertEqual(len(error_list), 4)

        for unexpected_param_val_pair in existing_parameter_value_pairs:
            self.assertIn(
                f"FOUND unexpected parameter-value-pair in combination list: "
                f"{get_nice_paremter_value_pair_str(unexpected_param_val_pair)}",
                error_list,
            )

    def test_check_unexpected_parameter_value_pair_in_combination_list_mixed_entries(self):
        # some of the pairs are in the combination list
        error_output = io.StringIO()

        existing_parameter_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), BOOST: (BOOST, 1.82)}),
                OD({HOST_COMPILER: (GCC, 10), CMAKE: (CMAKE, 3.22)}),
            ]
        )

        not_parameter_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({DEVICE_COMPILER: (CLANG_CUDA, 16), CMAKE: (CMAKE, 3.23)}),
                OD({DEVICE_COMPILER: (CLANG, 16), BOOST: (UBUNTU, 22.04)}),
                OD({DEVICE_COMPILER: (GCC, 7), BOOST: (UBUNTU, 22.04)}),
            ]
        )

        parameter_value_pairs: List[ParameterValuePair] = (
            existing_parameter_value_pairs + not_parameter_value_pairs
        )
        random.shuffle(parameter_value_pairs)

        self.assertFalse(
            check_unexpected_parameter_value_pair_in_combination_list(
                self.handwritten_comb_list,
                parameter_value_pairs,
                error_output,
            )
        )

        error_list = error_output.getvalue().rstrip().split("\n")
        self.assertEqual(len(error_list), 2)

        for unexpected_param_val_pair in existing_parameter_value_pairs:
            self.assertIn(
                f"FOUND unexpected parameter-value-pair in combination list: "
                f"{get_nice_paremter_value_pair_str(unexpected_param_val_pair)}",
                error_list,
            )

    def test_unrestricted_covertable_generator(self):
        comb_list: CombinationList = []
        # pylance shows a warning, because it cannot determine the concrete type of a namedtuple,
        # which is returned by AllPairs
        all_pairs: List[Dict[Parameter, ParameterValue]] = make(
            factors=self.param_matrix
        )  # type: ignore

        for all_pair in all_pairs:
            comb_list.append(OD(all_pair))

        self.assertTrue(
            check_parameter_value_pair_in_combination_list(comb_list, self.expected_param_val_pairs)
        )
