# pylint: disable=missing-docstring
import unittest
from typing import List
from utils_test import parse_expected_val_pairs, create_diff_parameter_value_pairs

from bashi.utils import bi_filter, parse_value_version, parse_parameter_single, parse_combination
from bashi.types import ParameterValue, ParameterValueSingle, ParameterValuePair, Combination
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


class TestBiFilter(unittest.TestCase):
    def test_bi_filter(self):
        input_list: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                ((HOST_COMPILER, NVCC, 11.2), (DEVICE_COMPILER, NVCC, 11.2)),
                ((HOST_COMPILER, NVCC, 9), (DEVICE_COMPILER, HIPCC, 11.7)),
                ((HOST_COMPILER, NVCC, 11.1), (DEVICE_COMPILER, NVCC, 12.0)),
                ((CMAKE, 3.23), (BOOST, 1.83)),
                ((HOST_COMPILER, GCC, 10), (DEVICE_COMPILER, GCC, 10)),
                ((HOST_COMPILER, ICPX, "2023.2.0"), (DEVICE_COMPILER, NVCC, 12.0)),
                ((HOST_COMPILER, CLANG, 10), (DEVICE_COMPILER, GCC, 11)),
                ((ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2)),
                ((HOST_COMPILER, NVCC, 10.1), (DEVICE_COMPILER, GCC, 11)),
            ]
        )

        expected_result: List[ParameterValuePair] = sorted(
            parse_expected_val_pairs(
                [
                    ((HOST_COMPILER, NVCC, 11.2), (DEVICE_COMPILER, NVCC, 11.2)),
                    ((HOST_COMPILER, NVCC, 9), (DEVICE_COMPILER, HIPCC, 11.7)),
                    ((HOST_COMPILER, NVCC, 11.1), (DEVICE_COMPILER, NVCC, 12.0)),
                    ((HOST_COMPILER, NVCC, 10.1), (DEVICE_COMPILER, GCC, 11)),
                ]
            )
        )
        unexpected_result: List[ParameterValuePair] = sorted(
            parse_expected_val_pairs(
                [
                    ((CMAKE, 3.23), (BOOST, 1.83)),
                    ((HOST_COMPILER, GCC, 10), (DEVICE_COMPILER, GCC, 10)),
                    ((HOST_COMPILER, ICPX, "2023.2.0"), (DEVICE_COMPILER, NVCC, 12.0)),
                    ((HOST_COMPILER, CLANG, 10), (DEVICE_COMPILER, GCC, 11)),
                    ((ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2)),
                ]
            )
        )

        def filter_func(param_value_pair: ParameterValuePair) -> bool:
            # keep all pairs, where first parameter-value is a NVCC host-compiler
            return (
                param_value_pair.first.parameter == HOST_COMPILER
                and param_value_pair.first.parameterValue.name == NVCC
            )

        removed_elements: List[ParameterValuePair] = []
        bi_filter(input_list, removed_elements, filter_func)

        input_list.sort()
        removed_elements.sort()

        self.assertEqual(
            input_list,
            expected_result,
            create_diff_parameter_value_pairs(input_list, expected_result),
        )
        self.assertEqual(
            removed_elements,
            unexpected_result,
            create_diff_parameter_value_pairs(removed_elements, unexpected_result),
        )


class TestParameterParser(unittest.TestCase):
    def test_value_version_str(self):
        self.assertEqual(parse_value_version("3.14"), packaging.version.parse("3.14"))

    def test_value_version_int(self):
        self.assertEqual(parse_value_version(7), packaging.version.parse("7"))

    def test_value_version_float(self):
        self.assertEqual(parse_value_version(8.09), packaging.version.parse("8.09"))

    def test_value_version_version(self):
        self.assertEqual(
            parse_value_version(packaging.version.parse("45.34")), packaging.version.parse("45.34")
        )

    def test_parse_parameter_single_two_values(self):
        self.assertEqual(
            parse_parameter_single((CMAKE, 3.14)),
            ParameterValueSingle(CMAKE, ParameterValue(CMAKE, packaging.version.parse("3.14"))),
        )

        self.assertEqual(
            parse_parameter_single((UBUNTU, "22.04")),
            ParameterValueSingle(UBUNTU, ParameterValue(UBUNTU, packaging.version.parse("22.04"))),
        )

    def test_parse_parameter_single_three_values(self):
        self.assertEqual(
            parse_parameter_single((HOST_COMPILER, GCC, 7)),
            ParameterValueSingle(HOST_COMPILER, ParameterValue(GCC, packaging.version.parse("7"))),
        )

        clang_version = packaging.version.parse("17")
        self.assertEqual(
            parse_parameter_single((DEVICE_COMPILER, CLANG, clang_version)),
            ParameterValueSingle(DEVICE_COMPILER, ParameterValue(CLANG, clang_version)),
        )

    def test_parse_combination(self):
        self.assertEqual(len(parse_combination([])), 0)

        comb = parse_combination(
            [
                (HOST_COMPILER, GCC, 7),
                (CMAKE, 3.14),
                (UBUNTU, "22.04"),
                (DEVICE_COMPILER, CLANG, packaging.version.parse("17")),
            ]
        )

        self.assertEqual(len(comb), 4)

        expected_comb = Combination(
            {
                HOST_COMPILER: ParameterValue(GCC, packaging.version.parse(str(7))),
                CMAKE: ParameterValue(CMAKE, packaging.version.parse(str(3.14))),
                UBUNTU: ParameterValue(UBUNTU, packaging.version.parse("22.04")),
                DEVICE_COMPILER: ParameterValue(CLANG, packaging.version.parse("17")),
            }
        )

        self.assertEqual(comb, expected_comb)
