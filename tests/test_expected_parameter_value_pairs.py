# pylint: disable=missing-docstring
import unittest
from typing import List, Union, Tuple
from collections import OrderedDict
import io
import packaging.version as pkv

# allpairspy has no type hints
from allpairspy import AllPairs  # type: ignore
from bashi.types import (
    Parameter,
    ParameterValue,
    ValueName,
    ParameterValuePair,
    ParameterValueMatrix,
    Combination,
    CombinationList,
)
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.utils import (
    get_expected_parameter_value_pairs,
    check_parameter_value_pair_in_combination_list,
    remove_parameter_value_pair,
)


def parse_param_val(param_val: Tuple[ValueName, Union[str, int, float]]) -> ParameterValue:
    val_name, val_version = param_val
    return ParameterValue(val_name, pkv.parse(str(val_version)))


def parse_param_vals(
    param_vals: List[Tuple[ValueName, Union[str, int, float]]]
) -> List[ParameterValue]:
    parsed_list: List[ParameterValue] = []

    for param_val in param_vals:
        parsed_list.append(parse_param_val(param_val))

    return parsed_list


def parse_expected_val_pairs(
    input_list: List[OrderedDict[Parameter, Tuple[ValueName, Union[str, int, float]]]]
) -> List[ParameterValuePair]:
    expected_val_pairs: List[ParameterValuePair] = []

    for param_val_pair in input_list:
        tmp_entry: ParameterValuePair = OrderedDict()
        for param in param_val_pair:
            tmp_entry[param] = parse_param_val(param_val_pair[param])
        expected_val_pairs.append(tmp_entry)

    return expected_val_pairs


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

        cls.generated_parameter_value_pairs: List[
            ParameterValuePair
        ] = get_expected_parameter_value_pairs(cls.param_matrix)

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
    def test_remove_two_entry_parameter_value_pair(self):
        OD = OrderedDict
        ppv = parse_param_val

        expected_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )
        original_length = len(expected_param_value_pairs)

        # expects one or two entries
        self.assertRaises(
            RuntimeError,
            remove_parameter_value_pair,
            OD(),
            expected_param_value_pairs,
        )
        self.assertRaises(
            RuntimeError,
            remove_parameter_value_pair,
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 9)),
                    DEVICE_COMPILER: ppv((NVCC, 11.2)),
                    CMAKE: ppv((CMAKE, 3.23)),
                }
            ),
            expected_param_value_pairs,
        )

        self.assertFalse(
            remove_parameter_value_pair(
                OD({HOST_COMPILER: ppv((GCC, 9)), DEVICE_COMPILER: ppv((NVCC, 11.2))}),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length)

        self.assertTrue(
            remove_parameter_value_pair(
                OD({HOST_COMPILER: ppv((GCC, 10)), DEVICE_COMPILER: ppv((NVCC, 12.0))}),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length - 1)

        self.assertTrue(
            remove_parameter_value_pair(
                OD({CMAKE: ppv((CMAKE, 3.23)), BOOST: ppv((BOOST, 1.83))}),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length - 2)

    def test_remove_single_entry_parameter_value_pair(self):
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

        self.assertFalse(
            remove_parameter_value_pair(
                OD({HOST_COMPILER: ppv((GCC, 12))}),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length)

        self.assertFalse(
            remove_parameter_value_pair(
                OD({HOST_COMPILER: ppv((CLANG, 12))}),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length)

        self.assertFalse(
            remove_parameter_value_pair(
                OD({UBUNTU: ppv((UBUNTU, 20.04))}),
                expected_param_value_pairs,
            )
        )
        self.assertEqual(len(expected_param_value_pairs), original_length)

        self.assertTrue(
            remove_parameter_value_pair(
                OD({HOST_COMPILER: ppv((GCC, 9))}),
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
