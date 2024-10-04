# pylint: disable=missing-docstring
import unittest
from typing import List, Union, Tuple
from collections import OrderedDict as OD
from packaging.specifiers import SpecifierSet
import packaging.version as pkv
from copy import deepcopy
from itertools import product


from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValuePair
from bashi.utils import _create_version_range, remove_parameter_value_pairs_ranges

from utils_test import parse_expected_val_pairs, create_diff_parameter_value_pairs


def bool_map(length: int):
    return [ele for ele in product([True, False], repeat=length)]


class TestCreateVersionRange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_min_max_range_expected: List[Tuple[Union[int, float, str], bool]] = [
            ("5.0.0", False),
            ("5.0.2", True),
            ("5.1.0", True),
            (9, True),
            (9.2, True),
            (9.4, False),
            (10, False),
            ("7.0.0", True),
            (81, False),
            (3, False),
            (4.3, False),
            (10.01, False),
            ("7.0.1", True),
            ("9.9.9", False),
        ]

    def test_min_max_any_version_type(self):
        expected_range = SpecifierSet() & SpecifierSet()
        for inclusive_min, inclusive_max in [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]:
            self.assertEqual(
                expected_range,
                _create_version_range(ANY_VERSION, inclusive_min, ANY_VERSION, inclusive_max),
            )

    def test_min_max_any_version_match(self):
        v_range = _create_version_range(ANY_VERSION, True, ANY_VERSION, True)
        for ver in [1, 67, 4.5, 34.4, "1.0.0", "345.34", "0"]:
            self.assertTrue(pkv.parse(str(ver)) in v_range)

    def test_min_range_inclusive(self):
        v_range = _create_version_range(7, True, ANY_VERSION, True)

        for ver, expected in [
            (6, False),
            (7, True),
            (7.0, True),
            ("7.0.0", True),
            (8, True),
            (3, False),
            (4.3, False),
            (7.01, True),
            ("7.0.1", True),
            ("6.9.9", False),
        ]:
            self.assertEqual(pkv.parse(str(ver)) in v_range, expected, f"{ver} -> {expected}")

    def test_min_range_exclusive(self):
        v_range = _create_version_range(7, False, ANY_VERSION, True)

        for ver, expected in [
            (6, False),
            (7, False),
            (7.0, False),
            ("7.0.0", False),
            (8, True),
            (3, False),
            (4.3, False),
            (7.01, True),
            ("7.0.1", True),
            ("6.9.9", False),
        ]:
            self.assertEqual(pkv.parse(str(ver)) in v_range, expected, f"{ver} -> {expected}")

        v_range_2 = _create_version_range("7.0.1", False, ANY_VERSION, True)

        for ver, expected in [
            (6, False),
            (7, False),
            (7.0, False),
            ("7.0.0", False),
            (8, True),
            ("7.0.1", False),
            ("7.0.2", True),
            ("7.1.0", True),
        ]:
            self.assertEqual(pkv.parse(str(ver)) in v_range_2, expected, f"{ver} -> {expected}")

    def test_max_range_inclusive(self):
        v_range = _create_version_range(ANY_VERSION, True, 13.1, True)

        for ver, expected in [
            (13, True),
            (13.1, True),
            (13.2, False),
            ("7.0.0", True),
            (8, True),
            (3, True),
            (44.3, False),
            (7.01, True),
            ("17.0.1", False),
            ("13.1.1", False),
        ]:
            self.assertEqual(pkv.parse(str(ver)) in v_range, expected, f"{ver} -> {expected}")

    def test_max_range_exclusive(self):
        v_range = _create_version_range(ANY_VERSION, True, 9, False)

        for ver, expected in [
            (8, True),
            (9, False),
            (10, False),
            ("7.0.0", True),
            (81, False),
            (3, True),
            (4.3, True),
            (10.01, False),
            ("7.0.1", True),
            ("9.9.9", False),
        ]:
            self.assertEqual(pkv.parse(str(ver)) in v_range, expected, f"{ver} -> {expected}")

    def test_min_max_range(self):
        for inclusive_min, inclusive_max in [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]:
            v_range = _create_version_range("5.0.1", inclusive_min, 9.3, inclusive_max)

            test_range_expected: List[Tuple[Union[int, float, str], bool]] = []
            test_range_expected[:] = self.test_min_max_range_expected
            test_range_expected.append(("5.0.1", inclusive_min))
            test_range_expected.append(
                (9.3, inclusive_max),
            )

            for ver, expected in test_range_expected:
                self.assertEqual(
                    pkv.parse(str(ver)) in v_range,
                    expected,
                    f"\ninclusive_min: {inclusive_min}, inclusive_max: {inclusive_max}\n"
                    f"{ver} -> {expected}",
                )


class TestRemoveParameterValuePairsRanges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameter_test_data: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                OD(
                    {
                        HOST_COMPILER: (GCC, 10),
                        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: (
                            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                            ON,
                        ),
                    }
                ),
                OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                OD({DEVICE_COMPILER: (HIPCC, 6.2), HOST_COMPILER: (HIPCC, 6.2)}),
            ]
        )
        # reuse data for parameter-value-names
        cls.name_test_data = cls.parameter_test_data

        cls.version_test_data: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                OD({DEVICE_COMPILER: (HIPCC, 6.2), HOST_COMPILER: (HIPCC, 6.2)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.5)}),
                # corner case GCC min version
                OD({HOST_COMPILER: (GCC, 8), DEVICE_COMPILER: (NVCC, 11.5)}),
                OD({HOST_COMPILER: (GCC, 7), DEVICE_COMPILER: (NVCC, 11.5)}),
                OD({HOST_COMPILER: (GCC, 12), DEVICE_COMPILER: (NVCC, 11.5)}),
                # corner case GCC max version
                OD({HOST_COMPILER: (GCC, 13), DEVICE_COMPILER: (NVCC, 11.5)}),
                OD({HOST_COMPILER: (GCC, 14), DEVICE_COMPILER: (NVCC, 11.5)}),
                OD({HOST_COMPILER: (GCC, 4), DEVICE_COMPILER: (NVCC, 11.5)}),
                OD({HOST_COMPILER: (GCC, 4), DEVICE_COMPILER: (NVCC, 10.2)}),
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.2)}),
                # corner case NVCC min version
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.3)}),
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.4)}),
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.6)}),
                # corner case NVCC max version
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.7)}),
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.8)}),
            ]
        )

    def test_remove_parameter_value_pairs_ranges_all_any(self):
        test_parameter_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                OD(
                    {
                        HOST_COMPILER: (GCC, 10),
                        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: (
                            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                            ON,
                        ),
                    }
                ),
                OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
            ]
        )

        expected_removed: List[ParameterValuePair] = []
        expected_removed[:] = sorted(test_parameter_value_pairs)

        remove_list: List[ParameterValuePair] = []
        remove_parameter_value_pairs_ranges(test_parameter_value_pairs, remove_list)

        test_parameter_value_pairs.sort()
        remove_list.sort()

        self.assertEqual(test_parameter_value_pairs, [])
        self.assertEqual(remove_list, expected_removed)

    def test_remove_parameter_value_pairs_ranges_remove_first_parameter(self):
        symmetric_test_data = deepcopy(self.parameter_test_data)
        symmetric_expected_result = sorted(
            parse_expected_val_pairs(
                [
                    OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                    OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                    OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                ]
            )
        )
        symmetric_expected_remove = sorted(
            list(set(symmetric_test_data) - set(symmetric_expected_result))
        )
        symmetric_remove: List[ParameterValuePair] = []

        remove_parameter_value_pairs_ranges(
            symmetric_test_data, symmetric_remove, parameter1=HOST_COMPILER
        )

        symmetric_test_data.sort()
        symmetric_remove.sort()

        self.assertEqual(
            symmetric_test_data,
            symmetric_expected_result,
            create_diff_parameter_value_pairs(symmetric_test_data, symmetric_expected_result),
        )
        self.assertEqual(
            symmetric_remove,
            symmetric_expected_remove,
            create_diff_parameter_value_pairs(symmetric_remove, symmetric_expected_remove),
        )

        non_symmetric_test_data = deepcopy(self.parameter_test_data)
        non_symmetric_expected_result = sorted(
            parse_expected_val_pairs(
                [
                    OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                    OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                    OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                    OD({DEVICE_COMPILER: (HIPCC, 6.2), HOST_COMPILER: (HIPCC, 6.2)}),
                ]
            )
        )
        non_symmetric_expected_remove = sorted(
            list(set(non_symmetric_test_data) - set(non_symmetric_expected_result))
        )
        non_symmetric_remove: List[ParameterValuePair] = []

        remove_parameter_value_pairs_ranges(
            non_symmetric_test_data, non_symmetric_remove, parameter1=HOST_COMPILER, symmetric=False
        )

        non_symmetric_test_data.sort()
        non_symmetric_remove.sort()

        self.assertEqual(
            non_symmetric_test_data,
            non_symmetric_expected_result,
            create_diff_parameter_value_pairs(
                non_symmetric_test_data, non_symmetric_expected_result
            ),
        )
        self.assertEqual(
            non_symmetric_remove,
            non_symmetric_expected_remove,
            create_diff_parameter_value_pairs(non_symmetric_remove, non_symmetric_expected_remove),
        )

    def test_remove_parameter_value_pairs_ranges_remove_second_parameter(self):
        symmetric_test_data = deepcopy(self.parameter_test_data)
        symmetric_expected_result = sorted(
            parse_expected_val_pairs(
                [
                    OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                    OD(
                        {
                            HOST_COMPILER: (GCC, 10),
                            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: (
                                ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                                ON,
                            ),
                        }
                    ),
                    OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                ]
            )
        )
        symmetric_expected_remove = sorted(
            list(set(symmetric_test_data) - set(symmetric_expected_result))
        )
        symmetric_remove: List[ParameterValuePair] = []

        remove_parameter_value_pairs_ranges(
            symmetric_test_data, symmetric_remove, parameter2=DEVICE_COMPILER
        )

        symmetric_test_data.sort()
        symmetric_remove.sort()

        self.assertEqual(
            symmetric_test_data,
            symmetric_expected_result,
            create_diff_parameter_value_pairs(symmetric_test_data, symmetric_expected_result),
        )
        self.assertEqual(
            symmetric_remove,
            symmetric_expected_remove,
            create_diff_parameter_value_pairs(symmetric_remove, symmetric_expected_remove),
        )

        non_symmetric_test_data = deepcopy(self.parameter_test_data)
        non_symmetric_expected_result = sorted(
            parse_expected_val_pairs(
                [
                    OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                    OD(
                        {
                            HOST_COMPILER: (GCC, 10),
                            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: (
                                ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                                ON,
                            ),
                        }
                    ),
                    OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                    OD({DEVICE_COMPILER: (HIPCC, 6.2), HOST_COMPILER: (HIPCC, 6.2)}),
                ]
            )
        )
        non_symmetric_expected_remove = sorted(
            list(set(non_symmetric_test_data) - set(non_symmetric_expected_result))
        )
        non_symmetric_remove: List[ParameterValuePair] = []

        remove_parameter_value_pairs_ranges(
            non_symmetric_test_data,
            non_symmetric_remove,
            parameter2=DEVICE_COMPILER,
            symmetric=False,
        )

        non_symmetric_test_data.sort()
        non_symmetric_remove.sort()

        self.assertEqual(
            non_symmetric_test_data,
            non_symmetric_expected_result,
            create_diff_parameter_value_pairs(
                non_symmetric_test_data, non_symmetric_expected_result
            ),
        )
        self.assertEqual(
            non_symmetric_remove,
            non_symmetric_expected_remove,
            create_diff_parameter_value_pairs(non_symmetric_remove, non_symmetric_expected_remove),
        )

    def test_remove_parameter_value_pairs_ranges_remove_both_parameter(self):
        test_data = deepcopy(self.parameter_test_data)
        expected_result = sorted(
            parse_expected_val_pairs(
                [
                    OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                    OD(
                        {
                            HOST_COMPILER: (GCC, 10),
                            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: (
                                ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                                ON,
                            ),
                        }
                    ),
                    OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                    OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                ]
            )
        )
        expected_remove = sorted(list(set(test_data) - set(expected_result)))
        remove: List[ParameterValuePair] = []

        remove_parameter_value_pairs_ranges(
            test_data, remove, parameter1=HOST_COMPILER, parameter2=DEVICE_COMPILER
        )

        test_data.sort()
        remove.sort()

        self.assertEqual(
            test_data,
            expected_result,
            create_diff_parameter_value_pairs(test_data, expected_result),
        )
        self.assertEqual(
            remove,
            expected_remove,
            create_diff_parameter_value_pairs(remove, expected_remove),
        )

    def test_remove_parameter_value_pairs_ranges_remove_single_name(self):
        test_data = deepcopy(self.name_test_data)
        expected_result = sorted(
            parse_expected_val_pairs(
                [
                    OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                    OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                    OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                    OD({DEVICE_COMPILER: (HIPCC, 6.2), HOST_COMPILER: (HIPCC, 6.2)}),
                ]
            )
        )
        expected_remove = sorted(list(set(test_data) - set(expected_result)))
        remove: List[ParameterValuePair] = []

        remove_parameter_value_pairs_ranges(
            test_data, remove, parameter1=HOST_COMPILER, value_name1=GCC
        )

        test_data.sort()
        remove.sort()

        self.assertEqual(
            test_data,
            expected_result,
            create_diff_parameter_value_pairs(test_data, expected_result),
        )
        self.assertEqual(
            remove,
            expected_remove,
            create_diff_parameter_value_pairs(remove, expected_remove),
        )

        name2_test_data = deepcopy(self.name_test_data)
        name2_remove: List[ParameterValuePair] = []

        remove_parameter_value_pairs_ranges(
            name2_test_data, name2_remove, parameter2=HOST_COMPILER, value_name2=GCC
        )
        name2_test_data.sort()
        name2_remove.sort()

        self.assertEqual(
            name2_test_data,
            test_data,
            create_diff_parameter_value_pairs(name2_test_data, test_data),
        )
        self.assertEqual(
            name2_remove,
            remove,
            create_diff_parameter_value_pairs(name2_remove, remove),
        )

    def test_remove_parameter_value_pairs_ranges_remove_both_name(self):
        test_data = deepcopy(self.name_test_data)
        expected_result = sorted(
            parse_expected_val_pairs(
                [
                    OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                    OD(
                        {
                            HOST_COMPILER: (GCC, 10),
                            ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: (
                                ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                                ON,
                            ),
                        }
                    ),
                    OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                    OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                    OD({DEVICE_COMPILER: (HIPCC, 6.2), HOST_COMPILER: (HIPCC, 6.2)}),
                ]
            )
        )
        expected_remove = sorted(list(set(test_data) - set(expected_result)))
        remove: List[ParameterValuePair] = []

        remove_parameter_value_pairs_ranges(
            test_data,
            remove,
            parameter1=HOST_COMPILER,
            value_name1=GCC,
            parameter2=DEVICE_COMPILER,
            value_name2=NVCC,
        )

        test_data.sort()
        remove.sort()

        self.assertEqual(
            test_data,
            expected_result,
            create_diff_parameter_value_pairs(test_data, expected_result),
        )
        self.assertEqual(
            remove,
            expected_remove,
            create_diff_parameter_value_pairs(remove, expected_remove),
        )

    # remove all elements up to a minimum version
    def test_remove_parameter_value_pairs_ranges_single_version_min_open(self):
        for inclusive_max1 in (True, False):
            test_data = deepcopy(self.version_test_data)
            expected_result = parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                    OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                    OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                    OD({DEVICE_COMPILER: (HIPCC, 6.2), HOST_COMPILER: (HIPCC, 6.2)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 12), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 13), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 14), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.3)}),
                    OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.4)}),
                    OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.6)}),
                    OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.7)}),
                    OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.8)}),
                ]
            )

            if not inclusive_max1:
                expected_result += parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (GCC, 8), DEVICE_COMPILER: (NVCC, 11.5)}),
                    ]
                )

            expected_result.sort()
            expected_remove = sorted(list(set(test_data) - set(expected_result)))
            remove: List[ParameterValuePair] = []

            remove_parameter_value_pairs_ranges(
                test_data,
                remove,
                parameter1=HOST_COMPILER,
                value_name1=GCC,
                value_max_version1=8,
                value_max_version1_inclusive=inclusive_max1,
            )

            test_data.sort()
            remove.sort()

            self.assertEqual(
                test_data,
                expected_result,
                f"\ninclusive_min1: {inclusive_max1}\n"
                + create_diff_parameter_value_pairs(test_data, expected_result),
            )
            self.assertEqual(
                remove,
                expected_remove,
                f"\ninclusive_min1: {inclusive_max1}\n"
                + create_diff_parameter_value_pairs(remove, expected_remove),
            )

    # remove all elements which are bigger than a minimum version
    def test_remove_parameter_value_pairs_ranges_single_version_max_open(self):
        for inclusive_min1 in (True, False):
            test_data = deepcopy(self.version_test_data)
            expected_result = parse_expected_val_pairs(
                [
                    OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                    OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                    OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                    OD({DEVICE_COMPILER: (HIPCC, 6.2), HOST_COMPILER: (HIPCC, 6.2)}),
                    OD({HOST_COMPILER: (GCC, 7), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 4), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 4), DEVICE_COMPILER: (NVCC, 10.2)}),
                ]
            )

            if not inclusive_min1:
                expected_result += parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (GCC, 8), DEVICE_COMPILER: (NVCC, 11.5)}),
                    ]
                )

            expected_result.sort()
            expected_remove = sorted(list(set(test_data) - set(expected_result)))
            remove: List[ParameterValuePair] = []

            remove_parameter_value_pairs_ranges(
                test_data,
                remove,
                parameter1=HOST_COMPILER,
                value_name1=GCC,
                value_min_version1=8,
                value_min_version1_inclusive=inclusive_min1,
            )

            test_data.sort()
            remove.sort()

            self.assertEqual(
                test_data,
                expected_result,
                f"\ninclusive_min1: {inclusive_min1}\n"
                + create_diff_parameter_value_pairs(test_data, expected_result),
            )
            self.assertEqual(
                remove,
                expected_remove,
                f"\ninclusive_min1: {inclusive_min1}\n"
                + create_diff_parameter_value_pairs(remove, expected_remove),
            )

    # remove all elements within a version range
    def test_remove_parameter_value_pairs_ranges_single_version_in_range(self):
        for inclusive_min1, inclusive_max1 in bool_map(2):
            test_data = deepcopy(self.version_test_data)
            expected_result = parse_expected_val_pairs(
                [
                    OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                    OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                    OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                    OD({DEVICE_COMPILER: (HIPCC, 6.2), HOST_COMPILER: (HIPCC, 6.2)}),
                    OD({HOST_COMPILER: (GCC, 7), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 14), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 4), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 4), DEVICE_COMPILER: (NVCC, 10.2)}),
                ]
            )

            if not inclusive_min1:
                expected_result += parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (GCC, 8), DEVICE_COMPILER: (NVCC, 11.5)}),
                    ]
                )

            if not inclusive_max1:
                expected_result += parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (GCC, 13), DEVICE_COMPILER: (NVCC, 11.5)}),
                    ]
                )

            expected_result.sort()
            expected_remove = sorted(list(set(test_data) - set(expected_result)))
            remove: List[ParameterValuePair] = []

            remove_parameter_value_pairs_ranges(
                test_data,
                remove,
                parameter1=HOST_COMPILER,
                value_name1=GCC,
                value_min_version1=8,
                value_min_version1_inclusive=inclusive_min1,
                value_max_version1=13,
                value_max_version1_inclusive=inclusive_max1,
            )

            test_data.sort()
            remove.sort()

            self.assertEqual(
                test_data,
                expected_result,
                f"\ninclusive_min1: {inclusive_min1} - inclusive_max1: {inclusive_max1}\n"
                + create_diff_parameter_value_pairs(test_data, expected_result),
            )
            self.assertEqual(
                remove,
                expected_remove,
                f"\ninclusive_min1: {inclusive_min1} - inclusive_max1: {inclusive_max1}\n"
                + create_diff_parameter_value_pairs(remove, expected_remove),
            )

    # remove all GCC 9 and newer (corner case GCC 8 ) and
    # remove all NVCC 11.6 and older (corner case NVCC 11.7)
    def test_remove_parameter_value_pairs_ranges_both_version_open(self):
        for inclusive_min1, inclusive_max2 in bool_map(2):
            test_data = deepcopy(self.version_test_data)
            expected_result = parse_expected_val_pairs(
                [
                    OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                    OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                    OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                    OD({DEVICE_COMPILER: (HIPCC, 6.2), HOST_COMPILER: (HIPCC, 6.2)}),
                    OD({HOST_COMPILER: (GCC, 7), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 4), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 4), DEVICE_COMPILER: (NVCC, 10.2)}),
                    OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.8)}),
                ]
            )

            if not inclusive_min1:
                expected_result += parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (GCC, 8), DEVICE_COMPILER: (NVCC, 11.5)}),
                    ]
                )

            if not inclusive_max2:
                expected_result += parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.7)}),
                    ]
                )

            expected_result.sort()
            expected_remove = sorted(list(set(test_data) - set(expected_result)))
            remove: List[ParameterValuePair] = []

            remove_parameter_value_pairs_ranges(
                test_data,
                remove,
                parameter1=HOST_COMPILER,
                value_name1=GCC,
                value_min_version1=8,
                value_min_version1_inclusive=inclusive_min1,
                parameter2=DEVICE_COMPILER,
                value_name2=NVCC,
                value_max_version2="11.7",
                value_max_version2_inclusive=inclusive_max2,
            )

            test_data.sort()
            remove.sort()

            self.assertEqual(
                test_data,
                expected_result,
                f"\ninclusive_min1: {inclusive_min1} - inclusive_max2: {inclusive_max2}\n"
                + create_diff_parameter_value_pairs(test_data, expected_result),
            )
            self.assertEqual(
                remove,
                expected_remove,
                f"\ninclusive_min1: {inclusive_min1} - inclusive_max2: {inclusive_max2}\n"
                + create_diff_parameter_value_pairs(remove, expected_remove),
            )

    def test_remove_parameter_value_pairs_ranges_both_version_in_range(self):
        for inclusive_min1, inclusive_max1, inclusive_min2, inclusive_max2 in bool_map(4):
            test_data = deepcopy(self.version_test_data)
            expected_result = parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({UBUNTU: (UBUNTU, "20.04"), CMAKE: (CMAKE, "3.19")}),
                    OD({BOOST: (BOOST, "1.81.1"), DEVICE_COMPILER: (GCC, 11)}),
                    OD({CMAKE: (CMAKE, "3.19"), UBUNTU: (UBUNTU, 3.19)}),
                    OD({DEVICE_COMPILER: (HIPCC, 6.2), HOST_COMPILER: (HIPCC, 6.2)}),
                    OD({HOST_COMPILER: (GCC, 7), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 14), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 4), DEVICE_COMPILER: (NVCC, 11.5)}),
                    OD({HOST_COMPILER: (GCC, 4), DEVICE_COMPILER: (NVCC, 10.2)}),
                    OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.8)}),
                ]
            )

            if not inclusive_min1:
                expected_result += parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (GCC, 8), DEVICE_COMPILER: (NVCC, 11.5)}),
                    ]
                )

            if not inclusive_max1:
                expected_result += parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (GCC, 13), DEVICE_COMPILER: (NVCC, 11.5)}),
                    ]
                )

            if not inclusive_min2:
                expected_result += parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.3)}),
                    ]
                )

            if not inclusive_max2:
                expected_result += parse_expected_val_pairs(
                    [
                        OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.7)}),
                    ]
                )

            expected_result.sort()
            expected_remove = sorted(list(set(test_data) - set(expected_result)))
            remove: List[ParameterValuePair] = []

            remove_parameter_value_pairs_ranges(
                test_data,
                remove,
                parameter1=HOST_COMPILER,
                value_name1=GCC,
                value_min_version1=8,
                value_min_version1_inclusive=inclusive_min1,
                value_max_version1=13,
                value_max_version1_inclusive=inclusive_max1,
                parameter2=DEVICE_COMPILER,
                value_name2=NVCC,
                value_min_version2=11.3,
                value_min_version2_inclusive=inclusive_min2,
                value_max_version2=11.7,
                value_max_version2_inclusive=inclusive_max2,
            )

            test_data.sort()
            remove.sort()

            self.assertEqual(
                test_data,
                expected_result,
                f"\ninclusive_min1: {inclusive_min1} - inclusive_max1: {inclusive_max1}\n"
                f"inclusive_min2: {inclusive_min2} - inclusive_max2: {inclusive_max2}\n"
                + create_diff_parameter_value_pairs(test_data, expected_result),
            )
            self.assertEqual(
                remove,
                expected_remove,
                f"\ninclusive_min1: {inclusive_min1} - inclusive_max1: {inclusive_max1}\n"
                f"inclusive_min2: {inclusive_min2} - inclusive_max2: {inclusive_max2}\n"
                + create_diff_parameter_value_pairs(remove, expected_remove),
            )
