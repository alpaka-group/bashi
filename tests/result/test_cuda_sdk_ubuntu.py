# pylint: disable=missing-docstring
# pylint: disable=too-many-lines
import unittest
from typing import Dict, Callable
from packaging.version import Version
from bashi.types import ParameterValuePair
from utils_test import (
    parse_expected_val_pairs2,
    default_remove_test,
    parse_value_version,
)
from bashi.runtime_info import ValidUbuntuSDK
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import CUDA_MIN_UBUNTU

from bashi.result_modules.cuda_support import (
    _remove_unsupported_nvcc_ubuntu_combinations,
    _remove_unsupported_cuda_backend_ubuntu_combinations,
    _remove_unsupported_clang_cuda_ubuntu_combinations,
    _remove_runtime_unsupported_cuda_backend_ubuntu_combinations,
    _remove_runtime_unsupported_clang_cuda_ubuntu_combinations,
)


class TestCUDASDKUbuntuStaticInfo(unittest.TestCase):
    def test_remove_unsupported_nvcc_ubuntu_combinations(self):
        latest_support_ubuntu_version: Version = sorted(CUDA_MIN_UBUNTU)[-1].ubuntu

        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((DEVICE_COMPILER, NVCC, 9.0), (UBUNTU, "18.04")),
                ((DEVICE_COMPILER, NVCC, 10.0), (UBUNTU, "18.04")),
                ((DEVICE_COMPILER, NVCC, 10.1), (UBUNTU, "20.04")),
                ((DEVICE_COMPILER, NVCC, 10.1), (UBUNTU, "22.04")),
                ((DEVICE_COMPILER, NVCC, 10.2), (UBUNTU, "24.04")),
                ((UBUNTU, "18.04"), (DEVICE_COMPILER, NVCC, 11.0)),
                ((UBUNTU, "20.04"), (DEVICE_COMPILER, NVCC, 11.4)),
                ((UBUNTU, "22.04"), (DEVICE_COMPILER, NVCC, 11.7)),
                ((UBUNTU, "24.04"), (DEVICE_COMPILER, NVCC, 11.9)),
                ((DEVICE_COMPILER, NVCC, 12.0), (UBUNTU, "20.04")),
                ((DEVICE_COMPILER, NVCC, 12.0), (UBUNTU, "24.04")),
                ((DEVICE_COMPILER, NVCC, 99.0), (UBUNTU, latest_support_ubuntu_version)),
            ]
        )

        expected_results: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((DEVICE_COMPILER, NVCC, 10.0), (UBUNTU, "18.04")),
                ((UBUNTU, "20.04"), (DEVICE_COMPILER, NVCC, 11.4)),
                ((DEVICE_COMPILER, NVCC, 12.0), (UBUNTU, "24.04")),
                ((DEVICE_COMPILER, NVCC, 99.0), (UBUNTU, latest_support_ubuntu_version)),
            ]
        )

        default_remove_test(
            _remove_unsupported_nvcc_ubuntu_combinations,
            test_param_value_pairs,
            expected_results,
            self,
        )

    def test_remove_unsupported_cuda_backend_ubuntu_combinations(self):
        latest_support_ubuntu_version: Version = sorted(CUDA_MIN_UBUNTU)[-1].ubuntu

        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "10.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "18.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "20.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "22.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "24.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 9.0), (UBUNTU, "18.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.0), (UBUNTU, "18.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.1), (UBUNTU, "20.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.1), (UBUNTU, "22.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.2), (UBUNTU, "24.04")),
                ((UBUNTU, "18.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.0)),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4)),
                ((UBUNTU, "22.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.7)),
                ((UBUNTU, "24.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.9)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0), (UBUNTU, "20.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0), (UBUNTU, "24.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 99.0), (UBUNTU, latest_support_ubuntu_version)),
            ]
        )

        expected_results: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "10.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "18.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "20.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "22.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "24.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.0), (UBUNTU, "18.04")),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0), (UBUNTU, "24.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 99.0), (UBUNTU, latest_support_ubuntu_version)),
            ]
        )

        default_remove_test(
            _remove_unsupported_cuda_backend_ubuntu_combinations,
            test_param_value_pairs,
            expected_results,
            self,
        )

    def test_remove_unsupported_clang_cuda_ubuntu_combinations(self):
        latest_support_ubuntu_version: Version = sorted(CUDA_MIN_UBUNTU)[-1].ubuntu

        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((DEVICE_COMPILER, CLANG_CUDA, 7), (UBUNTU, "18.04")),
                ((DEVICE_COMPILER, CLANG_CUDA, 7), (UBUNTU, "20.04")),
                ((DEVICE_COMPILER, CLANG_CUDA, 7), (UBUNTU, "24.04")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "18.04")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "20.04")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "24.04")),
                ((UBUNTU, "18.04"), (DEVICE_COMPILER, CLANG_CUDA, 15)),
                ((UBUNTU, "20.04"), (DEVICE_COMPILER, CLANG_CUDA, 15)),
                ((UBUNTU, "24.04"), (DEVICE_COMPILER, CLANG_CUDA, 15)),
                ((HOST_COMPILER, CLANG_CUDA, 18), (UBUNTU, "18.04")),
                ((HOST_COMPILER, CLANG_CUDA, 18), (UBUNTU, "20.04")),
                ((HOST_COMPILER, CLANG_CUDA, 18), (UBUNTU, "24.04")),
                ((HOST_COMPILER, CLANG_CUDA, 18), (UBUNTU, "26.04")),
                ((UBUNTU, latest_support_ubuntu_version), (HOST_COMPILER, CLANG_CUDA, 99)),
            ]
        )

        expected_results: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((DEVICE_COMPILER, CLANG_CUDA, 7.0), (UBUNTU, "18.04")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "18.04")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "20.04")),
                ((UBUNTU, "18.04"), (DEVICE_COMPILER, CLANG_CUDA, 15)),
                ((UBUNTU, "20.04"), (DEVICE_COMPILER, CLANG_CUDA, 15)),
                ((HOST_COMPILER, CLANG_CUDA, 18), (UBUNTU, "18.04")),
                ((HOST_COMPILER, CLANG_CUDA, 18), (UBUNTU, "20.04")),
                ((HOST_COMPILER, CLANG_CUDA, 18), (UBUNTU, "24.04")),
                ((UBUNTU, latest_support_ubuntu_version), (HOST_COMPILER, CLANG_CUDA, 99)),
            ]
        )

        default_remove_test(
            _remove_unsupported_clang_cuda_ubuntu_combinations,
            test_param_value_pairs,
            expected_results,
            self,
        )


class TestCUDASDKUbuntuRuntimeInfo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_input: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, CLANG, 23), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "4.08")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "18.04")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "20.04")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "24.04")),
                ((UBUNTU, "4.08"), (HOST_COMPILER, CLANG_CUDA, 17)),
                ((UBUNTU, "18.04"), (HOST_COMPILER, CLANG_CUDA, 17)),
                ((UBUNTU, "20.04"), (HOST_COMPILER, CLANG_CUDA, 17)),
                ((UBUNTU, "24.04"), (HOST_COMPILER, CLANG_CUDA, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "4.08")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "18.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "20.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "24.04")),
                ((UBUNTU, "4.08"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                ((UBUNTU, "18.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                ((UBUNTU, "24.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                ((UBUNTU, "4.08"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2)),
                ((UBUNTU, "18.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2)),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2)),
                ((UBUNTU, "24.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2)),
            ]
        )

    def test_remove_runtime_unsupported_cuda_backend_ubuntu_combinations_empty_runtime(self):
        default_remove_test(
            _remove_runtime_unsupported_cuda_backend_ubuntu_combinations,
            self.default_input,
            self.default_input,
            self,
            {},
        )

    def test__remove_runtime_unsupported_cuda_backend_ubuntu_combinations_normal_runtime_info(self):
        runtime_info: Dict[str, Callable[..., bool]] = {}
        runtime_info[RT_AVAILABLE_CUDA_SDK_UBUNTU_VER] = ValidUbuntuSDK(
            parse_value_version(["20.04", "22.04"])
        )

        expected_output = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, CLANG, 23), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "4.08")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "18.04")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "20.04")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "24.04")),
                ((UBUNTU, "4.08"), (HOST_COMPILER, CLANG_CUDA, 17)),
                ((UBUNTU, "18.04"), (HOST_COMPILER, CLANG_CUDA, 17)),
                ((UBUNTU, "20.04"), (HOST_COMPILER, CLANG_CUDA, 17)),
                ((UBUNTU, "24.04"), (HOST_COMPILER, CLANG_CUDA, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "4.08")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "18.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "20.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "24.04")),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2)),
            ]
        )
        default_remove_test(
            _remove_runtime_unsupported_cuda_backend_ubuntu_combinations,
            self.default_input,
            expected_output,
            self,
            runtime_info,
        )

    def test_remove_runtime_unsupported_clang_cuda_ubuntu_combinations_empty_runtime(self):
        default_remove_test(
            _remove_runtime_unsupported_clang_cuda_ubuntu_combinations,
            self.default_input,
            self.default_input,
            self,
            {},
        )

    def test_remove_runtime_unsupported_clang_cuda_ubuntu_combinations_normal_runtime_info(self):
        runtime_info: Dict[str, Callable[..., bool]] = {}
        runtime_info[RT_AVAILABLE_CUDA_SDK_UBUNTU_VER] = ValidUbuntuSDK(
            parse_value_version(["18.04", "20.04"])
        )

        expected_output = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, CLANG, 23), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "18.04")),
                ((HOST_COMPILER, CLANG_CUDA, 12), (UBUNTU, "20.04")),
                ((UBUNTU, "18.04"), (HOST_COMPILER, CLANG_CUDA, 17)),
                ((UBUNTU, "20.04"), (HOST_COMPILER, CLANG_CUDA, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "4.08")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "18.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "20.04")),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (UBUNTU, "24.04")),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2)),
            ]
        )
        default_remove_test(
            _remove_runtime_unsupported_clang_cuda_ubuntu_combinations,
            self.default_input,
            expected_output,
            self,
            runtime_info,
        )
