# pylint: disable=missing-docstring
import unittest
import os
import io
import copy
from collections import OrderedDict
import packaging.version as pkv
from utils_test import parse_param_vals
from bashi.versions import get_parameter_value_matrix
from bashi.generator import generate_combination_list
from bashi.utils import (
    check_parameter_value_pair_in_combination_list,
    remove_parameter_value_pairs,
)
from bashi.results import get_expected_bashi_parameter_value_pairs
from bashi.types import (
    ParameterValue,
    ParameterValuePair,
    ParameterValueTuple,
    ParameterValueMatrix,
)
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


class TestGeneratorTestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.param_matrix: ParameterValueMatrix = OrderedDict()

        cls.param_matrix[HOST_COMPILER] = parse_param_vals(
            [(GCC, 10), (GCC, 11), (GCC, 12), (CLANG, 16), (CLANG, 17)]
        )
        cls.param_matrix[DEVICE_COMPILER] = parse_param_vals(
            [
                (NVCC, 11.2),
                (NVCC, 12.0),
                (GCC, 10),
                (GCC, 11),
                (GCC, 12),
                (CLANG, 16),
                (CLANG, 17),
            ]
        )
        cls.param_matrix[CMAKE] = parse_param_vals([(CMAKE, 3.22), (CMAKE, 3.23)])
        cls.param_matrix[BOOST] = parse_param_vals([(BOOST, 1.81), (BOOST, 1.82), (BOOST, 1.83)])

        cls.generated_parameter_value_pairs: List[ParameterValuePair] = (
            get_expected_bashi_parameter_value_pairs(cls.param_matrix)
        )

    def test_generator_without_custom_filter(self):
        comb_list = generate_combination_list(self.param_matrix)

        self.assertTrue(
            check_parameter_value_pair_in_combination_list(
                comb_list, self.generated_parameter_value_pairs
            )
        )

    def test_generator_with_custom_filter(self):
        def custom_filter(row: ParameterValueTuple) -> bool:
            if DEVICE_COMPILER in row and row[DEVICE_COMPILER].name == NVCC:
                return False

            if (
                CMAKE in row
                and row[CMAKE].version == pkv.parse("3.23")
                and BOOST in row
                and row[BOOST].version == pkv.parse("1.82")
            ):
                return False

            return True

        OD = OrderedDict

        for row in [
            OD(
                {
                    HOST_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                    DEVICE_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                    DEVICE_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                    CMAKE: ParameterValue(CMAKE, pkv.parse("3.23")),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                    DEVICE_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                    BOOST: ParameterValue(BOOST, pkv.parse("1.82")),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                    DEVICE_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                    CMAKE: ParameterValue(CMAKE, pkv.parse("3.22")),
                    BOOST: ParameterValue(BOOST, pkv.parse("1.81")),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                    DEVICE_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                    CMAKE: ParameterValue(CMAKE, pkv.parse("3.23")),
                    BOOST: ParameterValue(BOOST, pkv.parse("1.81")),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                    DEVICE_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                    CMAKE: ParameterValue(CMAKE, pkv.parse("3.22")),
                    BOOST: ParameterValue(BOOST, pkv.parse("1.82")),
                }
            ),
        ]:
            self.assertTrue(custom_filter(row))

        self.assertFalse(
            custom_filter(
                OrderedDict(
                    {
                        HOST_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                        DEVICE_COMPILER: ParameterValue(GCC, pkv.parse("10")),
                        CMAKE: ParameterValue(CMAKE, pkv.parse("3.23")),
                        BOOST: ParameterValue(BOOST, pkv.parse("1.82")),
                    }
                )
            )
        )

        comb_list = generate_combination_list(
            parameter_value_matrix=self.param_matrix, custom_filter=custom_filter
        )

        reduced_expected_param_val_pairs = copy.deepcopy(self.generated_parameter_value_pairs)
        for device_compiler in self.param_matrix[DEVICE_COMPILER]:
            if device_compiler.name == NVCC:
                self.assertTrue(
                    remove_parameter_value_pairs(
                        reduced_expected_param_val_pairs,
                        parameter1=DEVICE_COMPILER,
                        value_name1=NVCC,
                        value_version1=str(device_compiler.version),
                    )
                )

        self.assertTrue(
            remove_parameter_value_pairs(
                reduced_expected_param_val_pairs,
                parameter1=CMAKE,
                value_name1=CMAKE,
                value_version1="3.23",
                parameter2=BOOST,
                value_name2=BOOST,
                value_version2="1.82",
            )
        )

        missing_combinations = io.StringIO()

        try:
            self.assertTrue(
                check_parameter_value_pair_in_combination_list(
                    comb_list, reduced_expected_param_val_pairs, missing_combinations
                )
            )
        except AssertionError as e:
            # remove comment to print missing, valid pairs
            print(f"\n{missing_combinations.getvalue()}")
            raise e


# set the environment variable NOLONG to skip long running tests
@unittest.skipIf("NOLONG" in os.environ, "Skip long running test")
class TestGeneratorRealData(unittest.TestCase):
    def test_generator_without_custom_filter(self):
        param_val_matrix = get_parameter_value_matrix()
        expected_param_val_pairs = get_expected_bashi_parameter_value_pairs(param_val_matrix)

        comb_list = generate_combination_list(param_val_matrix)

        self.assertTrue(
            check_parameter_value_pair_in_combination_list(comb_list, expected_param_val_pairs)
        )

    def test_generator_with_custom_filter(self):
        def custom_filter(row: ParameterValueTuple) -> bool:
            if (
                CMAKE in row
                and row[CMAKE].version == pkv.parse("3.23")
                and BOOST in row
                and row[BOOST].version == pkv.parse("1.82")
            ):
                return False

            return True

        param_val_matrix = get_parameter_value_matrix()
        reduced_expected_param_val_pairs = get_expected_bashi_parameter_value_pairs(
            param_val_matrix
        )

        self.assertTrue(
            remove_parameter_value_pairs(
                reduced_expected_param_val_pairs,
                parameter1=CMAKE,
                value_name1=CMAKE,
                value_version1="3.23",
                parameter2=BOOST,
                value_name2=BOOST,
                value_version2="1.82",
            )
        )

        comb_list = generate_combination_list(
            parameter_value_matrix=param_val_matrix, custom_filter=custom_filter
        )

        missing_combinations = io.StringIO()

        try:
            self.assertTrue(
                check_parameter_value_pair_in_combination_list(
                    comb_list, reduced_expected_param_val_pairs, missing_combinations
                )
            )
        except AssertionError as e:
            # remove comment to display missing combinations
            missing_combinations_str = missing_combinations.getvalue()
            print(f"\n{missing_combinations_str}")
            number_of_combs = len(missing_combinations_str.split("\n"))
            print(f"\nnumber of missing combinations: {number_of_combs}")
            raise e


class TestParameterMatrixFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.param_base_matrix: ParameterValueMatrix = OrderedDict()

        cls.param_base_matrix[HOST_COMPILER] = parse_param_vals(
            [(GCC, 10), (GCC, 11), (GCC, 12), (CLANG, 16), (CLANG, 17)]
        )
        cls.param_base_matrix[DEVICE_COMPILER] = parse_param_vals(
            [
                (GCC, 10),
                (GCC, 11),
                (GCC, 12),
                (CLANG, 16),
                (CLANG, 17),
            ]
        )
        cls.param_base_matrix[CMAKE] = parse_param_vals([(CMAKE, 3.22), (CMAKE, 3.23)])
        cls.param_base_matrix[BOOST] = parse_param_vals(
            [(BOOST, 1.81), (BOOST, 1.82), (BOOST, 1.83)]
        )

    def test_nvcc_host_compiler_rule_c1(self):
        # test if generate_combination_list() correctly handles nvcc as host compiler
        param_matrix = copy.deepcopy(self.param_base_matrix)
        for nvcc_version in [11.2, 11.3, 11.8, 12.0]:
            param_matrix[HOST_COMPILER].append(ParameterValue(NVCC, pkv.parse(str(nvcc_version))))
            param_matrix[DEVICE_COMPILER].append(ParameterValue(NVCC, pkv.parse(str(nvcc_version))))
        param_matrix_before = copy.deepcopy(param_matrix)

        comb_list = generate_combination_list(param_matrix)

        # generate_combination_list should not modify the param_matrix
        self.assertEqual(param_matrix_before, param_matrix)

        self.assertTrue(
            check_parameter_value_pair_in_combination_list(
                comb_list, get_expected_bashi_parameter_value_pairs(param_matrix)
            )
        )

    def test_clang_cuda_old_versions_rule_c8(self):
        # test if generate_combination_list() correctly clang-cuda version 13 and older

        param_matrix = copy.deepcopy(self.param_base_matrix)
        for clang_cuda_version in [8, 13, 14, 17]:
            param_matrix[HOST_COMPILER].append(
                ParameterValue(CLANG_CUDA, pkv.parse(str(clang_cuda_version)))
            )
            param_matrix[DEVICE_COMPILER].append(
                ParameterValue(CLANG_CUDA, pkv.parse(str(clang_cuda_version)))
            )
        param_matrix_before = copy.deepcopy(param_matrix)

        comb_list = generate_combination_list(parameter_value_matrix=param_matrix)

        # generate_combination_list should not modify the param_matrix
        self.assertEqual(param_matrix_before, param_matrix)

        self.assertTrue(
            check_parameter_value_pair_in_combination_list(
                comb_list, get_expected_bashi_parameter_value_pairs(param_matrix)
            )
        )
