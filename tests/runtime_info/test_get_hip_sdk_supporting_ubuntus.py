# pylint: disable=missing-docstring
import unittest
import packaging.version as pkv
from packaging.specifiers import SpecifierSet
from bashi.versions import UbuntuHipMinMax
from bashi.runtime_info import get_hip_sdk_supporting_ubuntus
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
