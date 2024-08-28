# pylint: disable=missing-docstring
import unittest
from typing import List
from collections import OrderedDict as OD
from utils_test import parse_expected_val_pairs, create_diff_parameter_value_pairs

from bashi.utils import bi_filter
from bashi.types import ParameterValuePair
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


class TestBiFilter(unittest.TestCase):
    def test_bi_filter(self):
        input_list: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (NVCC, 11.2), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (NVCC, 9), DEVICE_COMPILER: (HIPCC, 11.7)}),
                OD({HOST_COMPILER: (NVCC, 11.1), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (GCC, 10)}),
                OD({HOST_COMPILER: (ICPX, "2023.2.0"), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (GCC, 11)}),
                OD(
                    {
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2),
                    }
                ),
                OD({HOST_COMPILER: (NVCC, 10.1), DEVICE_COMPILER: (GCC, 11)}),
            ]
        )

        expected_result: List[ParameterValuePair] = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (NVCC, 11.2), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (NVCC, 9), DEVICE_COMPILER: (HIPCC, 11.7)}),
                    OD({HOST_COMPILER: (NVCC, 11.1), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (NVCC, 10.1), DEVICE_COMPILER: (GCC, 11)}),
                ]
            )
        )
        unexpected_result: List[ParameterValuePair] = sorted(
            parse_expected_val_pairs(
                [
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (GCC, 10)}),
                    OD({HOST_COMPILER: (ICPX, "2023.2.0"), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (GCC, 11)}),
                    OD(
                        {
                            ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: (
                                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                                ON,
                            ),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2),
                        }
                    ),
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
