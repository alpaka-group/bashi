import unittest
from typing import List
import packaging.version as pkv
from packaging.specifiers import SpecifierSet
from bashi.versions import (
    ClangCudaSDKSupport,
    SDKUbuntuSupport,
    UbuntuSDKMinMax,
    _get_ubuntu_clang_cuda_sdk_support,
    _get_ubuntu_sdk_min_max,
)


class TestGetUbuntuClangCudaSdkSupport(unittest.TestCase):
    def test_get_ubuntu_clang_cuda_sdk_support_case1(self):
        clang_cuda_max_cuda_version: List[ClangCudaSDKSupport] = [
            ClangCudaSDKSupport("7", "9.2"),
            ClangCudaSDKSupport("8", "10.0"),
            ClangCudaSDKSupport("10", "10.1"),
            ClangCudaSDKSupport("12", "11.0"),
            ClangCudaSDKSupport("13", "11.2"),
            ClangCudaSDKSupport("14", "11.5"),
            ClangCudaSDKSupport("16", "11.8"),
            ClangCudaSDKSupport("17", "12.1"),
            ClangCudaSDKSupport("19", "12.4"),
        ]

        cuda_min_ubuntu: List[SDKUbuntuSupport] = [
            SDKUbuntuSupport("10.0", "18.04"),
            SDKUbuntuSupport("11.0", "20.04"),
            SDKUbuntuSupport("12.0", "24.04"),
        ]
        ubuntu_cuda_version_range: List[UbuntuSDKMinMax] = _get_ubuntu_sdk_min_max(cuda_min_ubuntu)

        result = _get_ubuntu_clang_cuda_sdk_support(
            ubuntu_cuda_version_range, clang_cuda_max_cuda_version
        )

        self.assertEqual(
            result,
            {
                pkv.parse("18.04"): SpecifierSet(">=7"),
                pkv.parse("20.04"): SpecifierSet(">=12"),
                pkv.parse("24.04"): SpecifierSet(">=17"),
            },
        )

    def test_get_ubuntu_clang_cuda_sdk_support_case2(self):
        clang_cuda_max_cuda_version: List[ClangCudaSDKSupport] = [
            ClangCudaSDKSupport("7", "9.2"),
            ClangCudaSDKSupport("8", "10.0"),
            ClangCudaSDKSupport("10", "10.1"),
            ClangCudaSDKSupport("12", "11.0"),
            ClangCudaSDKSupport("13", "11.2"),
            ClangCudaSDKSupport("14", "11.5"),
            ClangCudaSDKSupport("16", "11.8"),
            ClangCudaSDKSupport("17", "12.1"),
            ClangCudaSDKSupport("19", "12.4"),
        ]

        cuda_min_ubuntu: List[SDKUbuntuSupport] = [
            SDKUbuntuSupport("11.0", "20.04"),
        ]
        ubuntu_cuda_version_range: List[UbuntuSDKMinMax] = _get_ubuntu_sdk_min_max(cuda_min_ubuntu)

        result = _get_ubuntu_clang_cuda_sdk_support(
            ubuntu_cuda_version_range, clang_cuda_max_cuda_version
        )

        self.assertEqual(
            result,
            {
                pkv.parse("20.04"): SpecifierSet(">=7"),
            },
        )

    def test_get_ubuntu_clang_cuda_sdk_support_case3(self):
        clang_cuda_max_cuda_version: List[ClangCudaSDKSupport] = [
            ClangCudaSDKSupport("7", "9.2"),
            ClangCudaSDKSupport("8", "10.0"),
            ClangCudaSDKSupport("10", "10.1"),
            ClangCudaSDKSupport("12", "11.0"),
            ClangCudaSDKSupport("13", "11.2"),
            ClangCudaSDKSupport("14", "11.5"),
            ClangCudaSDKSupport("16", "11.8"),
        ]

        cuda_min_ubuntu: List[SDKUbuntuSupport] = [
            SDKUbuntuSupport("10.0", "18.04"),
            SDKUbuntuSupport("11.0", "20.04"),
            SDKUbuntuSupport("12.0", "24.04"),
        ]
        ubuntu_cuda_version_range: List[UbuntuSDKMinMax] = _get_ubuntu_sdk_min_max(cuda_min_ubuntu)

        result = _get_ubuntu_clang_cuda_sdk_support(
            ubuntu_cuda_version_range, clang_cuda_max_cuda_version
        )

        self.assertEqual(
            result,
            {
                pkv.parse("18.04"): SpecifierSet(">=7"),
                pkv.parse("20.04"): SpecifierSet(">=12"),
            },
        )

    def test_get_ubuntu_clang_cuda_sdk_support_case4(self):
        clang_cuda_max_cuda_version: List[ClangCudaSDKSupport] = [
            ClangCudaSDKSupport("7", "9.2"),
            ClangCudaSDKSupport("8", "10.0"),
            ClangCudaSDKSupport("10", "10.1"),
            ClangCudaSDKSupport("12", "11.0"),
            ClangCudaSDKSupport("13", "11.2"),
            ClangCudaSDKSupport("14", "11.5"),
            ClangCudaSDKSupport("16", "11.8"),
            ClangCudaSDKSupport("17", "12.1"),
            ClangCudaSDKSupport("19", "12.4"),
        ]

        cuda_min_ubuntu: List[SDKUbuntuSupport] = [
            SDKUbuntuSupport("10.0", "18.04"),
            SDKUbuntuSupport("11.0", "20.04"),
        ]
        ubuntu_cuda_version_range: List[UbuntuSDKMinMax] = _get_ubuntu_sdk_min_max(cuda_min_ubuntu)

        result = _get_ubuntu_clang_cuda_sdk_support(
            ubuntu_cuda_version_range, clang_cuda_max_cuda_version
        )

        self.assertEqual(
            result,
            {
                pkv.parse("18.04"): SpecifierSet(">=7"),
                pkv.parse("20.04"): SpecifierSet(">=12"),
            },
        )
