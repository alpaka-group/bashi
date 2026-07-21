# pylint: disable=missing-docstring
# pylint: disable=too-many-lines
from typing import List
import unittest

from bashi.globals import (
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
    ALPAKA_ACC_GPU_CUDA_ENABLE,
    ALPAKA_ACC_GPU_HIP_ENABLE,
    ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
    DEVICE_COMPILER,
    GCC,
    HIPCC,
    HOST_COMPILER,
    ICPX,
    NVCC,
    OFF,
    ON,
)
from bashi.result_modules.custom_verifier import (
    remove_unsupported_compiler_backend_combinations,
    remove_unsupported_backend_combinations,
)
from bashi.types import CompilerBackendCombination, ParameterValuePair

from utils_test import parse_expected_val_pairs, default_remove_test


class TestRemoveUnsupportedCompilerBackendsCombinations(unittest.TestCase):
    t1_all_compilers = [GCC, NVCC, ICPX]
    t1_all_backends = [
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_HIP_ENABLE,
        ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
    ]
    t1_allowed_backend_combinations: List[CompilerBackendCombination] = []
    t1_test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
        [
            ((HOST_COMPILER, GCC, 8), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((DEVICE_COMPILER, GCC, 8), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
            ((DEVICE_COMPILER, NVCC, 11.2), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
            ((DEVICE_COMPILER, NVCC, 13.3), (ALPAKA_ACC_GPU_CUDA_ENABLE, 13.3)),
            ((DEVICE_COMPILER, NVCC, 9.2), (ALPAKA_ACC_GPU_CUDA_ENABLE, 9.2)),
            ((DEVICE_COMPILER, NVCC, 11.6), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
        ]
    )
    t1_expected_results: List[ParameterValuePair] = parse_expected_val_pairs(
        [
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((DEVICE_COMPILER, NVCC, 11.6), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
        ]
    )

    def test_no_allowed_backends_t1(self):
        default_remove_test(
            lambda parameter_value_pairs, removed_parameter_value_pairs: remove_unsupported_compiler_backend_combinations(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                self.t1_all_compilers,
                self.t1_all_backends,
                self.t1_allowed_backend_combinations,
            ),
            self.t1_test_param_value_pairs,
            self.t1_expected_results,
            self,
        )

    t2_all_compilers = [GCC, NVCC, ICPX]
    t2_all_backends = [
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_HIP_ENABLE,
        ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
    ]
    t2_allowed_backend_combinations: List[CompilerBackendCombination] = [
        CompilerBackendCombination(GCC, GCC, [ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE]),
        CompilerBackendCombination(GCC, NVCC, [ALPAKA_ACC_GPU_CUDA_ENABLE]),
        CompilerBackendCombination(ICPX, ICPX, [ALPAKA_ACC_ONEAPI_FPGA_ENABLE]),
    ]
    t2_test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
        [
            ((HOST_COMPILER, GCC, 8), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((DEVICE_COMPILER, GCC, 9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
            ((DEVICE_COMPILER, GCC, 9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
            ((HOST_COMPILER, GCC, 8), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.6)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((HOST_COMPILER, GCC, 8), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
            ((DEVICE_COMPILER, NVCC, 11.2), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
            ((DEVICE_COMPILER, NVCC, 13.3), (ALPAKA_ACC_GPU_CUDA_ENABLE, 13.3)),
            ((DEVICE_COMPILER, NVCC, 9.2), (ALPAKA_ACC_GPU_CUDA_ENABLE, 9.2)),
            ((DEVICE_COMPILER, NVCC, 11.6), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((DEVICE_COMPILER, NVCC, 11.9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
            ((DEVICE_COMPILER, NVCC, 11.9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
        ]
    )
    t2_expected_results: List[ParameterValuePair] = parse_expected_val_pairs(
        [
            ((HOST_COMPILER, GCC, 8), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((DEVICE_COMPILER, GCC, 9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
            ((HOST_COMPILER, GCC, 8), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.6)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
            ((DEVICE_COMPILER, NVCC, 11.2), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
            ((DEVICE_COMPILER, NVCC, 13.3), (ALPAKA_ACC_GPU_CUDA_ENABLE, 13.3)),
            ((DEVICE_COMPILER, NVCC, 9.2), (ALPAKA_ACC_GPU_CUDA_ENABLE, 9.2)),
            ((DEVICE_COMPILER, NVCC, 11.6), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((DEVICE_COMPILER, NVCC, 11.9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
        ]
    )

    def test_no_overlapping_backends_t2(self):
        default_remove_test(
            lambda parameter_value_pairs, removed_parameter_value_pairs: remove_unsupported_compiler_backend_combinations(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                self.t2_all_compilers,
                self.t2_all_backends,
                self.t2_allowed_backend_combinations,
            ),
            self.t2_test_param_value_pairs,
            self.t2_expected_results,
            self,
        )

    t3_all_compilers = [GCC, NVCC, HIPCC, ICPX]
    t3_all_backends = [
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_HIP_ENABLE,
        ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
    ]
    t3_allowed_backend_combinations: List[CompilerBackendCombination] = [
        CompilerBackendCombination(GCC, GCC, [ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE]),
        CompilerBackendCombination(
            GCC, NVCC, [ALPAKA_ACC_GPU_CUDA_ENABLE, ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE]
        ),
        CompilerBackendCombination(ICPX, ICPX, [ALPAKA_ACC_ONEAPI_FPGA_ENABLE]),
        CompilerBackendCombination(
            HIPCC,
            HIPCC,
            [
                ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                ALPAKA_ACC_GPU_CUDA_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
                ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
            ],
        ),
    ]
    t3_test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
        [
            ((HOST_COMPILER, GCC, 8), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((DEVICE_COMPILER, GCC, 9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
            ((DEVICE_COMPILER, GCC, 9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
            ((HOST_COMPILER, GCC, 8), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.6)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((HOST_COMPILER, GCC, 8), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
            ((DEVICE_COMPILER, NVCC, 11.2), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
            ((DEVICE_COMPILER, NVCC, 13.3), (ALPAKA_ACC_GPU_CUDA_ENABLE, 13.3)),
            ((DEVICE_COMPILER, NVCC, 9.2), (ALPAKA_ACC_GPU_CUDA_ENABLE, 9.2)),
            ((DEVICE_COMPILER, NVCC, 11.6), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((DEVICE_COMPILER, NVCC, 10.6), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((DEVICE_COMPILER, NVCC, 10.6), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((DEVICE_COMPILER, NVCC, 11.9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
            ((DEVICE_COMPILER, NVCC, 11.9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((HOST_COMPILER, HIPCC, 7.2), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((HOST_COMPILER, HIPCC, 7.2), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((DEVICE_COMPILER, HIPCC, 6.3), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((DEVICE_COMPILER, HIPCC, 6.3), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4)),
            ((DEVICE_COMPILER, HIPCC, 6.8), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
            ((DEVICE_COMPILER, HIPCC, 6.8), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
            ((HOST_COMPILER, HIPCC, 8.9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
            ((HOST_COMPILER, HIPCC, 8.9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
        ]
    )
    t3_expected_results: List[ParameterValuePair] = parse_expected_val_pairs(
        [
            ((HOST_COMPILER, GCC, 8), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((DEVICE_COMPILER, GCC, 9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
            ((HOST_COMPILER, GCC, 8), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.6)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((HOST_COMPILER, GCC, 9), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
            ((DEVICE_COMPILER, NVCC, 11.2), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
            ((DEVICE_COMPILER, NVCC, 13.3), (ALPAKA_ACC_GPU_CUDA_ENABLE, 13.3)),
            ((DEVICE_COMPILER, NVCC, 9.2), (ALPAKA_ACC_GPU_CUDA_ENABLE, 9.2)),
            ((DEVICE_COMPILER, NVCC, 11.6), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((DEVICE_COMPILER, NVCC, 10.6), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((DEVICE_COMPILER, NVCC, 10.6), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((DEVICE_COMPILER, NVCC, 11.9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
            ((DEVICE_COMPILER, ICPX, "2026.0.0"), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((HOST_COMPILER, HIPCC, 7.2), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((HOST_COMPILER, HIPCC, 7.2), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((DEVICE_COMPILER, HIPCC, 6.3), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((DEVICE_COMPILER, HIPCC, 6.3), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4)),
            ((DEVICE_COMPILER, HIPCC, 6.8), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
            ((DEVICE_COMPILER, HIPCC, 6.8), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
            ((HOST_COMPILER, HIPCC, 8.9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
            ((HOST_COMPILER, HIPCC, 8.9), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
        ]
    )

    def test_overlapping_backends_t3(self):
        default_remove_test(
            lambda parameter_value_pairs, removed_parameter_value_pairs: remove_unsupported_compiler_backend_combinations(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                self.t3_all_compilers,
                self.t3_all_backends,
                self.t3_allowed_backend_combinations,
            ),
            self.t3_test_param_value_pairs,
            self.t3_expected_results,
            self,
        )


class TestRemoveUnsupportedBackendCombinations(unittest.TestCase):
    t1_all_backends = [
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_HIP_ENABLE,
        ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
    ]
    t1_allowed_backend_combinations: List[CompilerBackendCombination] = []
    t1_test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
        [
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF), (ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
            ((ALPAKA_ACC_GPU_HIP_ENABLE, ON), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
        ]
    )
    t1_expected_results: List[ParameterValuePair] = parse_expected_val_pairs(
        [
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF), (ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
        ]
    )

    def test_no_allowed_backends_t1(self):
        default_remove_test(
            lambda parameter_value_pairs, removed_parameter_value_pairs: remove_unsupported_backend_combinations(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                self.t1_all_backends,
                self.t1_allowed_backend_combinations,
            ),
            self.t1_test_param_value_pairs,
            self.t1_expected_results,
            self,
        )

    t2_all_backends = [
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_HIP_ENABLE,
        ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
    ]
    t2_allowed_backend_combinations: List[CompilerBackendCombination] = [
        CompilerBackendCombination(GCC, GCC, [ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE]),
        CompilerBackendCombination(GCC, NVCC, [ALPAKA_ACC_GPU_CUDA_ENABLE]),
        CompilerBackendCombination(ICPX, ICPX, [ALPAKA_ACC_ONEAPI_FPGA_ENABLE]),
    ]
    t2_test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
        [
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF), (ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
            ((ALPAKA_ACC_GPU_HIP_ENABLE, ON), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
        ]
    )
    t2_expected_results: List[ParameterValuePair] = parse_expected_val_pairs(
        [
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF), (ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
        ]
    )

    def test_no_overlapping_backends_t2(self):
        default_remove_test(
            lambda parameter_value_pairs, removed_parameter_value_pairs: remove_unsupported_backend_combinations(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                self.t2_all_backends,
                self.t2_allowed_backend_combinations,
            ),
            self.t2_test_param_value_pairs,
            self.t2_expected_results,
            self,
        )

    t3_all_backends = [
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_HIP_ENABLE,
        ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
    ]
    t3_allowed_backend_combinations: List[CompilerBackendCombination] = [
        CompilerBackendCombination(GCC, GCC, [ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE]),
        CompilerBackendCombination(
            GCC, NVCC, [ALPAKA_ACC_GPU_CUDA_ENABLE, ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE]
        ),
        CompilerBackendCombination(ICPX, ICPX, [ALPAKA_ACC_ONEAPI_FPGA_ENABLE]),
        CompilerBackendCombination(
            HIPCC,
            HIPCC,
            [
                ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
                ALPAKA_ACC_GPU_CUDA_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
                ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
            ],
        ),
    ]
    t3_test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs(
        [
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF), (ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
            ((ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((ALPAKA_ACC_GPU_HIP_ENABLE, ON), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((ALPAKA_ACC_GPU_HIP_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
            ((ALPAKA_ACC_GPU_HIP_ENABLE, ON), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
            ((ALPAKA_ACC_GPU_HIP_ENABLE, OFF), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
        ]
    )
    t3_expected_results: List[ParameterValuePair] = parse_expected_val_pairs(
        [
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
            ((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF), (ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
            ((ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF)),
            ((ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((ALPAKA_ACC_GPU_HIP_ENABLE, ON), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON)),
            ((ALPAKA_ACC_GPU_HIP_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, ON)),
            ((ALPAKA_ACC_GPU_HIP_ENABLE, ON), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, ON)),
            ((ALPAKA_ACC_GPU_HIP_ENABLE, OFF), (ALPAKA_ACC_ONEAPI_FPGA_ENABLE, OFF)),
        ]
    )

    def test_overlapping_backends_t3(self):
        default_remove_test(
            lambda parameter_value_pairs, removed_parameter_value_pairs: remove_unsupported_backend_combinations(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                self.t3_all_backends,
                self.t3_allowed_backend_combinations,
            ),
            self.t3_test_param_value_pairs,
            self.t3_expected_results,
            self,
        )
