# pylint: disable=missing-docstring
# pylint: disable=too-many-lines
import unittest

from typing import Dict, Callable
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValuePair
from utils_test import (
    parse_expected_val_pairs2,
    default_remove_test,
    parse_value_version,
)
from bashi.runtime_info import ValidUbuntuSDK
from bashi.result_modules.hip_support import (
    _remove_unsupported_hipcc_ubuntu_combinations,
    _remove_unsupported_hip_backend_ubuntu_combinations,
)


class TestHipSDKUbuntuStaticInfo(unittest.TestCase):
    def test_remove_unsupported_hipcc_ubuntu_combinations(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                # remove because Ubuntu 16.04 is not in the support list
                ((HOST_COMPILER, HIPCC, 4.0), (UBUNTU, "16.04")),
                ((HOST_COMPILER, HIPCC, 5.0), (UBUNTU, "18.04")),
                ((HOST_COMPILER, HIPCC, 5.0), (UBUNTU, "20.04")),
                ((DEVICE_COMPILER, HIPCC, 5.0), (UBUNTU, "22.04")),
                ((DEVICE_COMPILER, HIPCC, 5.3), (UBUNTU, "20.04")),
                ((HOST_COMPILER, HIPCC, 5.9), (UBUNTU, "20.04")),
                ((UBUNTU, "22.04"), (HOST_COMPILER, HIPCC, 6.0)),
                ((UBUNTU, "20.04"), (HOST_COMPILER, HIPCC, 6.0)),
                ((UBUNTU, "24.04"), (DEVICE_COMPILER, HIPCC, 6.0)),
                ((HOST_COMPILER, HIPCC, 6.2), (UBUNTU, "22.04")),
                ((HOST_COMPILER, HIPCC, 6.2), (UBUNTU, "24.04")),
                ((HOST_COMPILER, HIPCC, 6.3), (UBUNTU, "22.04")),
                ((HOST_COMPILER, HIPCC, 6.3), (UBUNTU, "24.04")),
                ((HOST_COMPILER, HIPCC, 6.4), (UBUNTU, "24.04")),
                ((HOST_COMPILER, HIPCC, 6.3), (UBUNTU, "30.04")),
            ]
        )

        expected_results: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, HIPCC, 5.0), (UBUNTU, "20.04")),
                ((DEVICE_COMPILER, HIPCC, 5.3), (UBUNTU, "20.04")),
                ((HOST_COMPILER, HIPCC, 5.9), (UBUNTU, "20.04")),
                ((UBUNTU, "22.04"), (HOST_COMPILER, HIPCC, 6.0)),
                ((HOST_COMPILER, HIPCC, 6.2), (UBUNTU, "22.04")),
                ((HOST_COMPILER, HIPCC, 6.3), (UBUNTU, "24.04")),
                ((HOST_COMPILER, HIPCC, 6.4), (UBUNTU, "24.04")),
            ]
        )

        default_remove_test(
            _remove_unsupported_hipcc_ubuntu_combinations,
            test_param_value_pairs,
            expected_results,
            self,
        )


class TestHipSDKUbuntuRuntimeInfo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_input: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, HIPCC, 4.0), (UBUNTU, "16.04")),
                ((ALPAKA_ACC_GPU_HIP_ENABLE, ON), (UBUNTU, "16.04")),
                ((UBUNTU, "18.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                ((UBUNTU, "18.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                ((ALPAKA_ACC_GPU_HIP_ENABLE, ON), (UBUNTU, "20.04")),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                ((UBUNTU, "22.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                ((UBUNTU, "24.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                ((UBUNTU, "26.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                ((UBUNTU, "28.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
            ]
        )

    def test_remove_unsupported_hip_backend_ubuntu_combinations_empty_runtime_info(self):
        default_remove_test(
            _remove_unsupported_hip_backend_ubuntu_combinations,
            self.default_input,
            self.default_input,
            self,
            {},
        )

    def test_remove_unsupported_hip_backend_ubuntu_combinations_normal_runtime_info(self):
        runtime_info: Dict[str, Callable[..., bool]] = {}
        runtime_info[RT_AVAILABLE_HIP_SDK_UBUNTU_VER] = ValidUbuntuSDK(
            parse_value_version(["20.04", "22.04"])
        )

        expected_output = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, HIPCC, 4.0), (UBUNTU, "16.04")),
                ((UBUNTU, "18.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                ((ALPAKA_ACC_GPU_HIP_ENABLE, ON), (UBUNTU, "20.04")),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                ((UBUNTU, "22.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
            ]
        )
        default_remove_test(
            _remove_unsupported_hip_backend_ubuntu_combinations,
            self.default_input,
            expected_output,
            self,
            runtime_info,
        )

    def test_remove_unsupported_hip_backend_ubuntu_combinations_gapped_runtime_info(self):
        runtime_info: Dict[str, Callable[..., bool]] = {}
        runtime_info[RT_AVAILABLE_HIP_SDK_UBUNTU_VER] = ValidUbuntuSDK(
            parse_value_version(["20.04", "22.04", "26.04"])
        )

        expected_output = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, HIPCC, 4.0), (UBUNTU, "16.04")),
                ((UBUNTU, "18.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                ((ALPAKA_ACC_GPU_HIP_ENABLE, ON), (UBUNTU, "20.04")),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                ((UBUNTU, "22.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                ((UBUNTU, "26.04"), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
            ]
        )
        default_remove_test(
            _remove_unsupported_hip_backend_ubuntu_combinations,
            self.default_input,
            expected_output,
            self,
            runtime_info,
        )
