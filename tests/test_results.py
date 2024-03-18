# pylint: disable=missing-docstring
import unittest
import copy
from typing import List
from collections import OrderedDict as OD
import packaging.version as pkv
from utils_test import parse_expected_val_pairs, create_diff_parameter_value_pairs


from bashi.types import ParameterValue, ParameterValueSingle, ParameterValuePair
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import

# pyright: reportPrivateUsage=false
from bashi.results import (
    _remove_unsupported_nvcc_host_compiler,
    _remove_different_compiler_names,
    _remove_different_compiler_versions,
    _remove_unsupported_nvcc_cuda_host_compiler_versions,
    _remove_nvcc_unsupported_gcc_versions,
    _remove_nvcc_unsupported_clang_versions,
    _remove_specific_nvcc_clang_combinations,
    _remove_unsupported_compiler_for_hip_backend,
    _remove_disabled_hip_backend_for_hipcc,
    _remove_enabled_sycl_backend_for_hipcc,
    _remove_enabled_cuda_backend_for_hipcc,
    _remove_enabled_cuda_backend_for_enabled_hip_backend,
    _remove_unsupported_compiler_for_sycl_backend,
    _remove_disabled_sycl_backend_for_icpx,
    _remove_enabled_hip_backend_for_icpx,
    _remove_enabled_cuda_backend_for_icpx,
    _remove_enabled_cuda_backend_for_enabled_sycl_backend,
    _remove_nvcc_and_cuda_version_not_same,
    _remove_cuda_sdk_unsupported_gcc_versions,
    _remove_cuda_sdk_unsupported_clang_versions,
    _remove_device_compiler_gcc_clang_enabled_cuda_backend,
    _remove_specific_cuda_clang_combinations,
)
from bashi.versions import NvccHostSupport, NVCC_GCC_MAX_VERSION


class TestExpectedBashiParameterValuesPairs(unittest.TestCase):
    def test_nvcc_host_compiler_names(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.0)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                OD({HOST_COMPILER: (HIPCC, 5.1), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (HIPCC, 6.0), DEVICE_COMPILER: (NVCC, 11.3)}),
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
                OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (ICPX, "2023.2.0"), DEVICE_COMPILER: (ICPX, "2023.2.0")}),
            ]
        )

        _remove_unsupported_nvcc_host_compiler(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.0)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (GCC, 10)}),
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
                    OD({HOST_COMPILER: (ICPX, "2023.2.0"), DEVICE_COMPILER: (ICPX, "2023.2.0")}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_different_compiler_names(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (CLANG_CUDA, 10)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (CLANG, 7)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                OD({HOST_COMPILER: (HIPCC, 5.1), DEVICE_COMPILER: (ICPX, "2024.2.1")}),
                OD({HOST_COMPILER: (HIPCC, 6.0), DEVICE_COMPILER: (NVCC, 11.3)}),
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
                OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (HIPCC, 4.3), DEVICE_COMPILER: (HIPCC, 5.7)}),
                OD({HOST_COMPILER: (ICPX, "2023.2.0"), DEVICE_COMPILER: (ICPX, "2023.2.0")}),
            ]
        )

        _remove_different_compiler_names(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (CLANG, 7)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                    OD({HOST_COMPILER: (HIPCC, 6.0), DEVICE_COMPILER: (NVCC, 11.3)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (GCC, 10)}),
                    OD({HOST_COMPILER: (ICPX, "2023.2.0"), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD(
                        {
                            ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: (
                                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                                ON,
                            ),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2),
                        }
                    ),
                    OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (HIPCC, 4.3), DEVICE_COMPILER: (HIPCC, 5.7)}),
                    OD({HOST_COMPILER: (ICPX, "2023.2.0"), DEVICE_COMPILER: (ICPX, "2023.2.0")}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_different_compiler_versions(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (CLANG_CUDA, 10)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (CLANG, 7)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                OD({HOST_COMPILER: (HIPCC, 5.3), DEVICE_COMPILER: (HIPCC, 5.3)}),
                OD({HOST_COMPILER: (HIPCC, 5.1), DEVICE_COMPILER: (ICPX, "2024.2.1")}),
                OD({HOST_COMPILER: (HIPCC, 6.0), DEVICE_COMPILER: (NVCC, 11.3)}),
                OD({HOST_COMPILER: (CLANG_CUDA, 6), DEVICE_COMPILER: (CLANG_CUDA, 5)}),
                OD({HOST_COMPILER: (GCC, 7), DEVICE_COMPILER: (GCC, 10)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (GCC, 10)}),
                OD({HOST_COMPILER: (ICPX, "2023.2.0"), DEVICE_COMPILER: (ICPX, "2022.0.0")}),
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
                OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (HIPCC, 4.3), DEVICE_COMPILER: (HIPCC, 5.7)}),
                OD({HOST_COMPILER: (ICPX, "2023.2.0"), DEVICE_COMPILER: (ICPX, "2023.2.0")}),
            ]
        )

        _remove_different_compiler_versions(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (CLANG_CUDA, 10)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                    OD({HOST_COMPILER: (HIPCC, 5.3), DEVICE_COMPILER: (HIPCC, 5.3)}),
                    OD({HOST_COMPILER: (HIPCC, 5.1), DEVICE_COMPILER: (ICPX, "2024.2.1")}),
                    OD({HOST_COMPILER: (HIPCC, 6.0), DEVICE_COMPILER: (NVCC, 11.3)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (GCC, 10)}),
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
                    OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (ICPX, "2023.2.0"), DEVICE_COMPILER: (ICPX, "2023.2.0")}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_specific_nvcc_clang_combinations(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.4)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.3)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.4)}),
                OD({HOST_COMPILER: (CLANG, 11), DEVICE_COMPILER: (NVCC, 11.4)}),
                OD({HOST_COMPILER: (CLANG, 18), DEVICE_COMPILER: (NVCC, 11.4)}),
                OD({HOST_COMPILER: (CLANG, 14), DEVICE_COMPILER: (NVCC, 11.5)}),
                OD({HOST_COMPILER: (CLANG, 13), DEVICE_COMPILER: (NVCC, 11.6)}),
                OD({HOST_COMPILER: (CLANG, 17), DEVICE_COMPILER: (NVCC, 10.1)}),
                OD({HOST_COMPILER: (CLANG, 7), DEVICE_COMPILER: (NVCC, 12.3)}),
                OD(
                    {
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: (
                            ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                            ON,
                        ),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_specific_nvcc_clang_combinations(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.4)}),
                    OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (CLANG, 13), DEVICE_COMPILER: (NVCC, 11.6)}),
                    OD({HOST_COMPILER: (CLANG, 17), DEVICE_COMPILER: (NVCC, 10.1)}),
                    OD({HOST_COMPILER: (CLANG, 7), DEVICE_COMPILER: (NVCC, 12.3)}),
                    OD(
                        {
                            ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: (
                                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                                ON,
                            ),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )


TEST_HOST_COMPILER: str = "test_host_compiler"


class TestExpectedBashiParameterValuesPairsNvccHostCompilerVersions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gcc_param_value_matrix: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 12), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (GCC, 17), DEVICE_COMPILER: (NVCC, 99.0)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 12), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 13), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.8)}),
                OD({HOST_COMPILER: (GCC, 12), DEVICE_COMPILER: (NVCC, 11.8)}),
                OD({HOST_COMPILER: (GCC, 12), DEVICE_COMPILER: (NVCC, 12.3)}),
                OD({HOST_COMPILER: (GCC, 13), DEVICE_COMPILER: (NVCC, 12.3)}),
                OD({HOST_COMPILER: (GCC, 7), DEVICE_COMPILER: (NVCC, 10.1)}),
                OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 10.1)}),
                OD({HOST_COMPILER: (HIPCC, 5.3), DEVICE_COMPILER: (HIPCC, 5.3)}),
                OD({HOST_COMPILER: (GCC, 7), DEVICE_COMPILER: (GCC, 10)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
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

        cls.test_host_compiler_param_value_pairs: List[ParameterValuePair] = []
        for param_value_pair in cls.gcc_param_value_matrix:
            if param_value_pair.first.parameterValue.name == GCC:
                cls.test_host_compiler_param_value_pairs.append(
                    ParameterValuePair(
                        ParameterValueSingle(
                            param_value_pair.first.parameter,
                            ParameterValue(
                                TEST_HOST_COMPILER, param_value_pair.first.parameterValue.version
                            ),
                        ),
                        param_value_pair.second,
                    )
                )
            else:
                cls.test_host_compiler_param_value_pairs.append(copy.deepcopy(param_value_pair))

        cls.test_host_compiler_param_value_pairs += parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.5)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 13), DEVICE_COMPILER: (NVCC, 12.5)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 14), DEVICE_COMPILER: (NVCC, 12.5)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.8)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 13), DEVICE_COMPILER: (NVCC, 12.8)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 14), DEVICE_COMPILER: (NVCC, 12.8)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 15), DEVICE_COMPILER: (NVCC, 12.8)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 16), DEVICE_COMPILER: (NVCC, 12.8)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.9)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 13), DEVICE_COMPILER: (NVCC, 12.9)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 14), DEVICE_COMPILER: (NVCC, 12.9)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 15), DEVICE_COMPILER: (NVCC, 12.9)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 16), DEVICE_COMPILER: (NVCC, 12.9)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 5), DEVICE_COMPILER: (NVCC, 10.0)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 6), DEVICE_COMPILER: (NVCC, 10.0)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 7), DEVICE_COMPILER: (NVCC, 10.0)}),
                OD({HOST_COMPILER: (TEST_HOST_COMPILER, 52), DEVICE_COMPILER: (NVCC, 9.2)}),
            ]
        )

        cls.test_host_compiler_support_list: List[NvccHostSupport] = [
            NvccHostSupport("12.0", "12"),
            NvccHostSupport("11.4", "11"),
            NvccHostSupport("11.1", "10"),
            NvccHostSupport("11.0", "9"),
            NvccHostSupport("10.1", "8"),
            NvccHostSupport("10.0", "6"),
        ]
        cls.test_host_compiler_support_list.sort(reverse=True)

    def test_remove_nvcc_unsupported_host_compiler_versions_same_compiler_version_end(self):
        test_param_value_pairs: List[ParameterValuePair] = copy.deepcopy(
            self.test_host_compiler_param_value_pairs
        )
        test_support_list = copy.deepcopy(self.test_host_compiler_support_list)
        test_support_list.append(NvccHostSupport("12.7", "13"))
        test_support_list.append(NvccHostSupport("12.8", "13"))

        _remove_unsupported_nvcc_cuda_host_compiler_versions(
            test_param_value_pairs, TEST_HOST_COMPILER, DEVICE_COMPILER, NVCC, test_support_list
        )

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 9), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 17), DEVICE_COMPILER: (NVCC, 99.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 11), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 11), DEVICE_COMPILER: (NVCC, 11.8)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.3)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 52), DEVICE_COMPILER: (NVCC, 9.2)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 5), DEVICE_COMPILER: (NVCC, 10.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 6), DEVICE_COMPILER: (NVCC, 10.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 7), DEVICE_COMPILER: (NVCC, 10.1)}),
                    OD({HOST_COMPILER: (HIPCC, 5.3), DEVICE_COMPILER: (HIPCC, 5.3)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 7), DEVICE_COMPILER: (GCC, 10)}),
                    OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
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
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.5)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.8)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 13), DEVICE_COMPILER: (NVCC, 12.8)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.9)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 13), DEVICE_COMPILER: (NVCC, 12.9)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 14), DEVICE_COMPILER: (NVCC, 12.9)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 15), DEVICE_COMPILER: (NVCC, 12.9)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 16), DEVICE_COMPILER: (NVCC, 12.9)}),
                ]
            )
        )
        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_nvcc_unsupported_host_compiler_versions_different_compiler_version_end(self):
        test_param_value_pairs: List[ParameterValuePair] = copy.deepcopy(
            self.test_host_compiler_param_value_pairs
        )
        test_support_list = copy.deepcopy(self.test_host_compiler_support_list)
        test_support_list.append(NvccHostSupport("12.7", "13"))
        test_support_list.append(NvccHostSupport("12.8", "14"))

        _remove_unsupported_nvcc_cuda_host_compiler_versions(
            test_param_value_pairs, TEST_HOST_COMPILER, DEVICE_COMPILER, NVCC, test_support_list
        )

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 9), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 17), DEVICE_COMPILER: (NVCC, 99.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 11), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 11), DEVICE_COMPILER: (NVCC, 11.8)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.3)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 7), DEVICE_COMPILER: (NVCC, 10.1)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 5), DEVICE_COMPILER: (NVCC, 10.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 6), DEVICE_COMPILER: (NVCC, 10.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 52), DEVICE_COMPILER: (NVCC, 9.2)}),
                    OD({HOST_COMPILER: (HIPCC, 5.3), DEVICE_COMPILER: (HIPCC, 5.3)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 7), DEVICE_COMPILER: (GCC, 10)}),
                    OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
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
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.5)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.8)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 13), DEVICE_COMPILER: (NVCC, 12.8)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 14), DEVICE_COMPILER: (NVCC, 12.8)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.9)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 13), DEVICE_COMPILER: (NVCC, 12.9)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 14), DEVICE_COMPILER: (NVCC, 12.9)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 15), DEVICE_COMPILER: (NVCC, 12.9)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 16), DEVICE_COMPILER: (NVCC, 12.9)}),
                ]
            )
        )
        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_nvcc_unsupported_host_compiler_versions_different_compiler_version_end2(self):
        test_param_value_pairs: List[ParameterValuePair] = copy.deepcopy(
            self.test_host_compiler_param_value_pairs
        )
        test_support_list = copy.deepcopy(self.test_host_compiler_support_list)
        test_support_list.append(NvccHostSupport("12.7", "13"))
        test_support_list.append(NvccHostSupport("12.8", "15"))

        _remove_unsupported_nvcc_cuda_host_compiler_versions(
            test_param_value_pairs, TEST_HOST_COMPILER, DEVICE_COMPILER, NVCC, test_support_list
        )

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 9), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 17), DEVICE_COMPILER: (NVCC, 99.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 11), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 11), DEVICE_COMPILER: (NVCC, 11.8)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.3)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 7), DEVICE_COMPILER: (NVCC, 10.1)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 5), DEVICE_COMPILER: (NVCC, 10.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 6), DEVICE_COMPILER: (NVCC, 10.0)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 52), DEVICE_COMPILER: (NVCC, 9.2)}),
                    OD({HOST_COMPILER: (HIPCC, 5.3), DEVICE_COMPILER: (HIPCC, 5.3)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 7), DEVICE_COMPILER: (GCC, 10)}),
                    OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
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
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.5)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.8)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 13), DEVICE_COMPILER: (NVCC, 12.8)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 14), DEVICE_COMPILER: (NVCC, 12.8)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 15), DEVICE_COMPILER: (NVCC, 12.8)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 12), DEVICE_COMPILER: (NVCC, 12.9)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 13), DEVICE_COMPILER: (NVCC, 12.9)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 14), DEVICE_COMPILER: (NVCC, 12.9)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 15), DEVICE_COMPILER: (NVCC, 12.9)}),
                    OD({HOST_COMPILER: (TEST_HOST_COMPILER, 16), DEVICE_COMPILER: (NVCC, 12.9)}),
                ]
            )
        )
        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_nvcc_unsupported_gcc_versions(self):
        supported_nvcc_versions = [ver.nvcc for ver in NVCC_GCC_MAX_VERSION]
        # we assume for the test, that the this nvcc versions are supported by bashi
        for nvcc_version in ["10.0", "10.1", "11.0", "11.1", "11.4", "12.0"]:
            self.assertIn(pkv.parse(nvcc_version), supported_nvcc_versions)

        # for the test, it is required that CUDA 99.0 is not supported
        self.assertFalse(pkv.parse("99.0") in supported_nvcc_versions)

        test_param_value_pairs = copy.deepcopy(self.gcc_param_value_matrix)

        _remove_nvcc_unsupported_gcc_versions(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (GCC, 17), DEVICE_COMPILER: (NVCC, 99.0)}),
                    OD({HOST_COMPILER: (GCC, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                    OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (GCC, 12), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (GCC, 11), DEVICE_COMPILER: (NVCC, 11.8)}),
                    OD({HOST_COMPILER: (GCC, 12), DEVICE_COMPILER: (NVCC, 12.3)}),
                    OD({HOST_COMPILER: (GCC, 7), DEVICE_COMPILER: (NVCC, 10.1)}),
                    OD({HOST_COMPILER: (HIPCC, 5.3), DEVICE_COMPILER: (HIPCC, 5.3)}),
                    OD({HOST_COMPILER: (GCC, 7), DEVICE_COMPILER: (GCC, 10)}),
                    OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
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

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_nvcc_unsupported_clang_versions(self):
        supported_nvcc_versions = [ver.nvcc for ver in NVCC_GCC_MAX_VERSION]
        # we assume for the test, that the this nvcc versions are supported by bashi
        for nvcc_version in ["10.0", "10.1", "11.0", "11.1", "11.4", "12.0"]:
            self.assertIn(pkv.parse(nvcc_version), supported_nvcc_versions)

        # for the test, it is required that CUDA 99.0 is not supported
        self.assertFalse(pkv.parse("99.0") in supported_nvcc_versions)

        test_param_value_pairs = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                OD({HOST_COMPILER: (CLANG, 9), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (CLANG, 11), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (CLANG, 12), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({HOST_COMPILER: (CLANG, 17), DEVICE_COMPILER: (NVCC, 99.0)}),
                OD({HOST_COMPILER: (CLANG, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                OD({HOST_COMPILER: (CLANG, 11), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (CLANG, 14), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (CLANG, 17), DEVICE_COMPILER: (NVCC, 12.0)}),
                OD({HOST_COMPILER: (CLANG, 12), DEVICE_COMPILER: (NVCC, 11.8)}),
                OD({HOST_COMPILER: (CLANG, 14), DEVICE_COMPILER: (NVCC, 11.8)}),
                OD({HOST_COMPILER: (CLANG, 15), DEVICE_COMPILER: (NVCC, 12.3)}),
                OD({HOST_COMPILER: (CLANG, 16), DEVICE_COMPILER: (NVCC, 12.3)}),
                OD({HOST_COMPILER: (CLANG, 17), DEVICE_COMPILER: (NVCC, 12.3)}),
                OD({HOST_COMPILER: (CLANG, 7), DEVICE_COMPILER: (NVCC, 10.1)}),
                OD({HOST_COMPILER: (CLANG, 6), DEVICE_COMPILER: (NVCC, 10.0)}),
                OD({HOST_COMPILER: (CLANG, 7), DEVICE_COMPILER: (NVCC, 10.0)}),
                OD({HOST_COMPILER: (CLANG, 9), DEVICE_COMPILER: (NVCC, 10.1)}),
                OD({HOST_COMPILER: (HIPCC, 5.3), DEVICE_COMPILER: (HIPCC, 5.3)}),
                OD({HOST_COMPILER: (GCC, 7), DEVICE_COMPILER: (GCC, 10)}),
                OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
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

        _remove_nvcc_unsupported_clang_versions(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD({HOST_COMPILER: (CLANG_CUDA, 16), DEVICE_COMPILER: (CLANG_CUDA, 16)}),
                    OD({HOST_COMPILER: (CLANG, 9), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (CLANG, 11), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (CLANG, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({HOST_COMPILER: (CLANG, 17), DEVICE_COMPILER: (NVCC, 99.0)}),
                    OD({HOST_COMPILER: (CLANG, 9), DEVICE_COMPILER: (NVCC, 11.7)}),
                    OD({HOST_COMPILER: (CLANG, 11), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (CLANG, 14), DEVICE_COMPILER: (NVCC, 12.0)}),
                    OD({HOST_COMPILER: (CLANG, 12), DEVICE_COMPILER: (NVCC, 11.8)}),
                    OD({HOST_COMPILER: (CLANG, 15), DEVICE_COMPILER: (NVCC, 12.3)}),
                    OD({HOST_COMPILER: (CLANG, 16), DEVICE_COMPILER: (NVCC, 12.3)}),
                    OD({HOST_COMPILER: (CLANG, 7), DEVICE_COMPILER: (NVCC, 10.1)}),
                    OD({HOST_COMPILER: (CLANG, 6), DEVICE_COMPILER: (NVCC, 10.0)}),
                    OD({HOST_COMPILER: (HIPCC, 5.3), DEVICE_COMPILER: (HIPCC, 5.3)}),
                    OD({HOST_COMPILER: (GCC, 7), DEVICE_COMPILER: (GCC, 10)}),
                    OD({HOST_COMPILER: (GCC, 10), DEVICE_COMPILER: (NVCC, 11.2)}),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
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

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )


class TestExpectedBashiParameterValuesPairsHIPBackend(unittest.TestCase):
    def test_remove_unsupported_compiler_for_hip_backend(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        HOST_COMPILER: (CLANG_CUDA, 16),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (GCC, 10),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (HIPCC, 5.1),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (ICPX, "2023.1.0"),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (HIPCC, 6.0),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (NVCC, 11.2),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_unsupported_compiler_for_hip_backend(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            HOST_COMPILER: (HIPCC, 5.1),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (HIPCC, 6.0),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_disabled_hip_backend_for_hipcc(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        HOST_COMPILER: (CLANG_CUDA, 16),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (GCC, 10),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (HIPCC, 5.1),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (ICPX, "2023.1.0"),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (HIPCC, 6.0),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (NVCC, 11.2),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_disabled_hip_backend_for_hipcc(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            HOST_COMPILER: (CLANG_CUDA, 16),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (GCC, 10),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (ICPX, "2023.1.0"),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (NVCC, 11.2),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_enabled_sycl_backend_for_hipcc(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        HOST_COMPILER: (CLANG_CUDA, 16),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (GCC, 10),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (HIPCC, 4.3),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (HIPCC, 5.1),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (ICPX, "2023.1.0"),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (HIPCC, 6.0),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (NVCC, 11.2),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                OD(
                    {
                        DEVICE_COMPILER: (HIPCC, 5.7),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                    }
                ),
            ]
        )

        _remove_enabled_sycl_backend_for_hipcc(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            HOST_COMPILER: (CLANG_CUDA, 16),
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (GCC, 10),
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (HIPCC, 5.1),
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (ICPX, "2023.1.0"),
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (HIPCC, 6.0),
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (NVCC, 11.2),
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_enabled_cuda_backend_for_hipcc(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        HOST_COMPILER: (CLANG_CUDA, 16),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 10.1),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (GCC, 10),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (HIPCC, 5.1),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (HIPCC, 5.3),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (ICPX, "2023.1.0"),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (HIPCC, 6.0),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (HIPCC, 6.1),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (NVCC, 11.2),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_enabled_cuda_backend_for_hipcc(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            HOST_COMPILER: (CLANG_CUDA, 16),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 10.1),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (GCC, 10),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (HIPCC, 5.1),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (ICPX, "2023.1.0"),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (HIPCC, 6.0),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (NVCC, 11.2),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_enabled_cuda_backend_for_enabled_hip_backend(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_enabled_cuda_backend_for_enabled_hip_backend(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )


class TestExpectedBashiParameterValuesPairsSYCLBackend(unittest.TestCase):
    def test_remove_unsupported_compiler_for_sycl_backend(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        HOST_COMPILER: (CLANG_CUDA, 16),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (GCC, 10),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (ICPX, "2023.1.0"),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (HIPCC, 5.1),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (ICPX, "2024.2.0"),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (NVCC, 11.2),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_unsupported_compiler_for_sycl_backend(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            HOST_COMPILER: (ICPX, "2023.1.0"),
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (ICPX, "2024.2.0"),
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_disabled_sycl_backend_for_icpx(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        HOST_COMPILER: (CLANG_CUDA, 16),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (GCC, 10),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (ICPX, "2023.1.0"),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (HIPCC, 5.1),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (ICPX, "2024.2.0"),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (NVCC, 11.2),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_disabled_sycl_backend_for_icpx(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            HOST_COMPILER: (CLANG_CUDA, 16),
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (GCC, 10),
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (HIPCC, 5.1),
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (NVCC, 11.2),
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_enabled_hip_backend_for_icpx(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        HOST_COMPILER: (CLANG_CUDA, 16),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (GCC, 10),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (ICPX, "2023.1.0"),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (ICPX, "2024.2.0"),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (HIPCC, 5.1),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (ICPX, "2024.2.1"),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (NVCC, 11.2),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                OD(
                    {
                        DEVICE_COMPILER: (HIPCC, 5.7),
                        ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                    }
                ),
            ]
        )

        _remove_enabled_hip_backend_for_icpx(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            HOST_COMPILER: (CLANG_CUDA, 16),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (GCC, 10),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (ICPX, "2023.1.0"),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (ICPX, "2024.2.0"),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (HIPCC, 5.1),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (NVCC, 11.2),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                    OD(
                        {
                            DEVICE_COMPILER: (HIPCC, 5.7),
                            ALPAKA_ACC_GPU_HIP_ENABLE: (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                        }
                    ),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_enabled_cuda_backend_for_icpx(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        HOST_COMPILER: (CLANG_CUDA, 16),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 10.1),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (GCC, 10),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (ICPX, "2023.1.0"),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (ICPX, "2024.2.0"),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (HIPCC, 4.5),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (ICPX, "2024.2.1"),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (ICPX, "2024.8.3"),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (NVCC, 11.2),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_enabled_cuda_backend_for_icpx(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            HOST_COMPILER: (CLANG_CUDA, 16),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 10.1),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (GCC, 10),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (ICPX, "2023.1.0"),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (HIPCC, 4.5),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (ICPX, "2024.2.1"),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (NVCC, 11.2),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_enabled_cuda_backend_for_enabled_sycl_backend(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                    }
                ),
                OD(
                    {
                        ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_enabled_cuda_backend_for_enabled_sycl_backend(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, ON),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            ALPAKA_ACC_SYCL_ENABLE: (ALPAKA_ACC_SYCL_ENABLE, OFF),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )


class TestExpectedBashiParameterValuesPairsNvccCudaBackend(unittest.TestCase):
    def test_remove_nvcc_and_cuda_version_not_same(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        DEVICE_COMPILER: (NVCC, 11.2),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (NVCC, 11.2),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (NVCC, 12.1),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.3),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (NVCC, 12.1),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_nvcc_and_cuda_version_not_same(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            DEVICE_COMPILER: (NVCC, 11.2),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (NVCC, 12.1),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_cuda_sdk_unsupported_gcc_versions(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        HOST_COMPILER: (GCC, 10),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (GCC, 12),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (ICPX, "2022.3"),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.3),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (GCC, 12),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_cuda_sdk_unsupported_gcc_versions(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            HOST_COMPILER: (GCC, 10),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (ICPX, "2022.3"),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.3),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (GCC, 12),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_cuda_sdk_unsupported_clang_versions(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        HOST_COMPILER: (CLANG, 10),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (CLANG, 15),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (ICPX, "2022.3"),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.3),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (CLANG, 12),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_cuda_sdk_unsupported_clang_versions(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            HOST_COMPILER: (CLANG, 10),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (ICPX, "2022.3"),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.3),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (CLANG, 12),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_device_compiler_gcc_clang_enabled_cuda_backend(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD(
                    {
                        DEVICE_COMPILER: (GCC, 10),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (CLANG, 15),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (GCC, 10),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (ICPX, "2022.3"),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.3),
                    }
                ),
                OD(
                    {
                        DEVICE_COMPILER: (CLANG, 12),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_device_compiler_gcc_clang_enabled_cuda_backend(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            DEVICE_COMPILER: (GCC, 10),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (ICPX, "2022.3"),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.3),
                        }
                    ),
                    OD(
                        {
                            DEVICE_COMPILER: (CLANG, 12),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )

    def test_remove_specific_cuda_clang_combinations(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
            [
                OD({HOST_COMPILER: (CLANG_CUDA, 16), ALPAKA_ACC_GPU_CUDA_ENABLE: (CLANG_CUDA, 16)}),
                OD(
                    {
                        HOST_COMPILER: (GCC, 10),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (CLANG, 10),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (CLANG, 10),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.3),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (CLANG, 10),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (CLANG, 11),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (CLANG, 18),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (CLANG, 14),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.5),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (CLANG, 13),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.6),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (CLANG, 17),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 10.1),
                    }
                ),
                OD(
                    {
                        HOST_COMPILER: (CLANG, 7),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.3),
                    }
                ),
                OD(
                    {
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: (
                            ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                            ON,
                        ),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2),
                    }
                ),
                OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
            ]
        )

        _remove_specific_cuda_clang_combinations(test_param_value_pairs)

        test_param_value_pairs.sort()
        expected_results = sorted(
            parse_expected_val_pairs(
                [
                    OD(
                        {
                            HOST_COMPILER: (CLANG_CUDA, 16),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (CLANG_CUDA, 16),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (GCC, 10),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (CLANG, 10),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (CLANG, 13),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.6),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (CLANG, 17),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 10.1),
                        }
                    ),
                    OD(
                        {
                            HOST_COMPILER: (CLANG, 7),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.3),
                        }
                    ),
                    OD(
                        {
                            ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: (
                                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                                ON,
                            ),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2),
                        }
                    ),
                    OD({CMAKE: (CMAKE, 3.23), BOOST: (BOOST, 1.83)}),
                ]
            )
        )

        self.assertEqual(
            test_param_value_pairs,
            expected_results,
            create_diff_parameter_value_pairs(test_param_value_pairs, expected_results),
        )
