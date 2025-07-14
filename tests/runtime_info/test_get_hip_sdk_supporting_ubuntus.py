# pylint: disable=missing-docstring
import unittest
import packaging.version as pkv
from packaging.specifiers import SpecifierSet
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import UbuntuHipMinMax, get_parameter_value_matrix, VERSIONS
from bashi.runtime_info import get_hip_sdk_supporting_ubuntus
from bashi.generator import get_runtime_infos
from collections import OrderedDict
from bashi.types import ParameterValue, ParameterValueMatrix
from utils_test import parse_value_version as pvv


class TestGetHipSdkSupportingUbuntus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_ubuntu_hip_version_range = [
            UbuntuHipMinMax(ubuntu=pkv.parse("18.04"), hip_range=SpecifierSet("<5.0")),
            UbuntuHipMinMax(ubuntu=pkv.parse("20.04"), hip_range=SpecifierSet(">=5.0, <6.0")),
            UbuntuHipMinMax(ubuntu=pkv.parse("22.04"), hip_range=SpecifierSet(">=6.0, <6.3")),
            UbuntuHipMinMax(ubuntu=pkv.parse("24.04"), hip_range=SpecifierSet(">=6.3")),
        ]

    def test_get_hip_sdk_supporting_ubuntus_empty_input(self):
        with self.assertRaises(RuntimeError):
            get_hip_sdk_supporting_ubuntus(ubuntus=[], hipccs=[], ubuntu_hip_version_range=[])

        with self.assertRaises(RuntimeError):
            get_hip_sdk_supporting_ubuntus(
                ubuntus=[],
                hipccs=[],
                ubuntu_hip_version_range=self.default_ubuntu_hip_version_range,
            )

        with self.assertRaises(RuntimeError):
            get_hip_sdk_supporting_ubuntus(
                ubuntus=[],
                hipccs=[pkv.parse("6.1")],
                ubuntu_hip_version_range=[],
            )

        with self.assertRaises(RuntimeError):
            get_hip_sdk_supporting_ubuntus(
                ubuntus=[pkv.parse("20.04")],
                hipccs=[],
                ubuntu_hip_version_range=[],
            )

    def test_get_hip_sdk_supporting_ubuntus_all_hip_available(self):
        given_ubuntus = pvv(["16.04", "18.04", "20.04", "22.04", "24.04", "26.04"])
        expected_ubuntus = pvv(["18.04", "20.04", "22.04", "24.04"])
        validator = get_hip_sdk_supporting_ubuntus(
            ubuntus=given_ubuntus,
            hipccs=pvv([4.0, 5.1, 6.1, 6.3]),
            ubuntu_hip_version_range=self.default_ubuntu_hip_version_range,
        )

        self.assertEqual(
            sorted(validator.valid_ubuntus),
            sorted(expected_ubuntus),
        )

        for ub in given_ubuntus:
            self.assertEqual(validator(ub), ub in expected_ubuntus)

    def test_get_hip_sdk_supporting_ubuntus_all_hip_available_edge_case_mapping(self):
        given_ubuntus = pvv(["16.04", "18.04", "20.04", "22.04", "24.04", "26.04"])
        expected_ubuntus = pvv(["18.04", "20.04"])
        validator = get_hip_sdk_supporting_ubuntus(
            ubuntus=given_ubuntus,
            hipccs=pvv([4.0, 5.1, 6.1, 6.3]),
            ubuntu_hip_version_range=[
                UbuntuHipMinMax(ubuntu=pkv.parse("18.04"), hip_range=SpecifierSet("<5.0")),
                UbuntuHipMinMax(ubuntu=pkv.parse("20.04"), hip_range=SpecifierSet(">=5.0")),
            ],
        )

        self.assertEqual(
            sorted(validator.valid_ubuntus),
            sorted(expected_ubuntus),
        )

        for ub in given_ubuntus:
            self.assertEqual(validator(ub), ub in expected_ubuntus)

    def test_get_hip_sdk_supporting_ubuntus_missing_hip_versions1(self):
        given_ubuntus = pvv(["18.04", "20.04", "22.04", "24.04"])
        expected_ubuntus = pvv(["18.04", "22.04"])
        validator = get_hip_sdk_supporting_ubuntus(
            ubuntus=given_ubuntus,
            hipccs=pvv([4.0, 6.1]),
            ubuntu_hip_version_range=self.default_ubuntu_hip_version_range,
        )

        self.assertEqual(
            sorted(validator.valid_ubuntus),
            sorted(expected_ubuntus),
        )

        for ub in given_ubuntus:
            self.assertEqual(validator(ub), ub in expected_ubuntus)

    def test_get_hip_sdk_supporting_ubuntus_missing_hip_versions2(self):
        given_ubuntus = pvv(["18.04", "20.04", "22.04", "24.04"])
        expected_ubuntus = pvv(["20.04", "22.04", "24.04"])
        validator = get_hip_sdk_supporting_ubuntus(
            ubuntus=given_ubuntus,
            hipccs=pvv([5.0, 6.1, 6.5]),
            ubuntu_hip_version_range=self.default_ubuntu_hip_version_range,
        )

        self.assertEqual(
            sorted(validator.valid_ubuntus),
            sorted(expected_ubuntus),
        )

        for ub in given_ubuntus:
            self.assertEqual(validator(ub), ub in expected_ubuntus)


class TestGetRuntimeInfoUbuntuHip(unittest.TestCase):
    def test_get_runtime_infos_ubuntu_hip_all_param_vals_available(self):
        parameter_value_matrix = get_parameter_value_matrix()
        runtime_info = get_runtime_infos(parameter_value_matrix)
        self.assertIn(RT_AVAILABLE_HIP_SDK_UBUNTU_VER, runtime_info)

    def test_get_runtime_infos_ubuntu_hip_ubuntu_missing(self):
        parameter_value_matrix = get_parameter_value_matrix()
        del parameter_value_matrix[UBUNTU]
        runtime_info = get_runtime_infos(parameter_value_matrix)
        self.assertNotIn(RT_AVAILABLE_HIP_SDK_UBUNTU_VER, runtime_info)

    def test_get_runtime_infos_ubuntu_hip_missing_hipcc(self):
        param_val_matrix: ParameterValueMatrix = OrderedDict()

        for compiler_type in [HOST_COMPILER, DEVICE_COMPILER]:
            param_val_matrix[compiler_type] = []
            for sw_name, sw_versions in VERSIONS.items():
                if sw_name != HIPCC:
                    if sw_name in COMPILERS:
                        for sw_version in sw_versions:
                            param_val_matrix[compiler_type].append(
                                ParameterValue(sw_name, pkv.parse(str(sw_version)))
                            )

        for backend in BACKENDS:
            if backend == ALPAKA_ACC_GPU_CUDA_ENABLE:
                param_val_matrix[backend] = [ParameterValue(backend, OFF_VER)]
                for cuda_version in VERSIONS[NVCC]:
                    param_val_matrix[backend].append(
                        ParameterValue(backend, pkv.parse(str(cuda_version)))
                    )
            else:
                param_val_matrix[backend] = [
                    ParameterValue(backend, OFF_VER),
                    ParameterValue(backend, ON_VER),
                ]

        for other, versions in VERSIONS.items():
            if not other in COMPILERS + BACKENDS:
                param_val_matrix[other] = []
                for version in versions:
                    param_val_matrix[other].append(ParameterValue(other, pkv.parse(str(version))))

        runtime_info = get_runtime_infos(param_val_matrix)
        self.assertNotIn(RT_AVAILABLE_HIP_SDK_UBUNTU_VER, runtime_info)
