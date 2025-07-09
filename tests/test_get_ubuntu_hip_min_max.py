import unittest
from typing import List
import packaging.version as pkv
from packaging.specifiers import SpecifierSet
from bashi.versions import HIPUbuntuSupport, UbuntuHipMinMax, _get_ubuntu_hip_min_max


class TestGetUbuntuHipMinMax(unittest.TestCase):
    def test_get_ubuntu_hip_min_max_empty_input(self):
        self.assertEqual(len(_get_ubuntu_hip_min_max([])), 0)

    def test_get_ubuntu_hip_min_max_normal_list(self):
        hip_min_ubuntu: List[HIPUbuntuSupport] = [
            HIPUbuntuSupport("5.0", "20.04"),
            HIPUbuntuSupport("6.0", "22.04"),
            HIPUbuntuSupport("6.3", "24.04"),
        ]

        self.assertEqual(
            _get_ubuntu_hip_min_max(hip_min_ubuntu),
            [
                UbuntuHipMinMax(pkv.parse("20.04"), SpecifierSet("<5.0")),
                UbuntuHipMinMax(pkv.parse("20.04"), SpecifierSet(">=5.0, <6.0")),
                UbuntuHipMinMax(pkv.parse("22.04"), SpecifierSet(">=6.0, <6.3")),
                UbuntuHipMinMax(pkv.parse("24.04"), SpecifierSet(">=6.3")),
            ],
        )

    def test_get_ubuntu_hip_min_max_single_entry(self):
        hip_min_ubuntu: List[HIPUbuntuSupport] = [
            HIPUbuntuSupport("6.0", "22.04"),
        ]

        self.assertEqual(
            _get_ubuntu_hip_min_max(hip_min_ubuntu),
            [
                UbuntuHipMinMax(pkv.parse("22.04"), SpecifierSet("<6.0")),
                UbuntuHipMinMax(pkv.parse("22.04"), SpecifierSet(">=6.0")),
            ],
        )

    def test_get_ubuntu_hip_min_max_dual_value(self):
        hip_min_ubuntu: List[HIPUbuntuSupport] = [
            HIPUbuntuSupport("5.0", "20.04"),
            HIPUbuntuSupport("6.0", "22.04"),
        ]

        self.assertEqual(
            _get_ubuntu_hip_min_max(hip_min_ubuntu),
            [
                UbuntuHipMinMax(pkv.parse("20.04"), SpecifierSet("<5.0")),
                UbuntuHipMinMax(pkv.parse("20.04"), SpecifierSet(">=5.0, <6.0")),
                UbuntuHipMinMax(pkv.parse("22.04"), SpecifierSet(">=6.0")),
            ],
        )
