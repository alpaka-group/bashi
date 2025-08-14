# pylint: disable=missing-docstring
import unittest
from typing import List, Union
import packaging.version as pkv
from bashi.versions import VERSIONS, is_supported_version
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.exceptions import BashiUnknownVersion


def parse_to_version(input_list: List[Union[int, float, str]]) -> List[pkv.Version]:
    parsed_versions: List[pkv.Version] = []

    for v in input_list:
        parsed_versions.append(pkv.parse(str(v)))

    return parsed_versions


class TestParameterValueGenerator(unittest.TestCase):
    def test_compiler_versions(self):
        # test if version is supported for the following tests
        for name, version in [(GCC, 12), (CLANG, 14), (NVCC, 12.1)]:
            self.assertTrue(
                pkv.parse(str(version)) in parse_to_version(VERSIONS[name]),
                f"{name} {version} is not supported by bashi",
            )

        # test if version is unsupported for the following tests
        for name, version in [(GCC, 1), (NVCC, 4.7)]:
            self.assertFalse(
                pkv.parse(str(version)) in parse_to_version(VERSIONS[name]),
                f"{name} {version} is supported by bashi",
            )

        self.assertTrue(is_supported_version(GCC, pkv.parse(str(12))))
        self.assertTrue(is_supported_version(CLANG, pkv.parse(str(14))))
        self.assertTrue(is_supported_version(CLANG_CUDA, pkv.parse(str(14))))
        self.assertTrue(is_supported_version(NVCC, pkv.parse(str(12.1))))

        self.assertFalse(is_supported_version(GCC, pkv.parse(str(1))))
        self.assertFalse(is_supported_version(NVCC, pkv.parse(str(4.7))))

        # unknown compiler should throw an error
        self.assertRaises(
            BashiUnknownVersion, is_supported_version, "fancy-cpp-compiler", pkv.parse(str(12))
        )

    def test_backend_versions(self):
        # all compilers except the CUDA backend has only the versions ON and OFF
        for backend in BACKENDS:
            if backend != ALPAKA_ACC_GPU_CUDA_ENABLE:
                self.assertTrue(is_supported_version(backend, ON_VER))
                self.assertTrue(is_supported_version(backend, OFF_VER))
                self.assertFalse(is_supported_version(backend, pkv.parse(str(12.1))))

        # test if version is supported for the following tests
        for name, version in [(NVCC, 12.1)]:
            self.assertTrue(
                pkv.parse(str(version)) in parse_to_version(VERSIONS[name]),
                f"{name} {version} is not supported by bashi",
            )

        # test if version is unsupported for the following tests
        for name, version in [(NVCC, ON_VER), (NVCC, 4.7)]:
            self.assertFalse(
                pkv.parse(str(version)) in parse_to_version(VERSIONS[name]),
                f"{name} {version} is supported by bashi",
            )

        self.assertTrue(is_supported_version(ALPAKA_ACC_GPU_CUDA_ENABLE, OFF_VER))
        self.assertFalse(is_supported_version(ALPAKA_ACC_GPU_CUDA_ENABLE, ON_VER))
        self.assertTrue(is_supported_version(ALPAKA_ACC_GPU_CUDA_ENABLE, pkv.parse(str(12.1))))

        # unknown backend should throw an error
        self.assertRaises(
            BashiUnknownVersion, is_supported_version, "alpaka_Backend_esoteric_acc", OFF_VER
        )

    def test_software_versions(self):
        for name, version in [
            (CMAKE, 3.19),
            (CMAKE, 3.21),
            (BOOST, "1.80.0"),
            (UBUNTU, "20.04"),
            (CXX_STANDARD, 17),
        ]:
            # test if version is supported for the following tests
            self.assertTrue(
                pkv.parse(str(version)) in parse_to_version(VERSIONS[name]),
                f"{name} {version} is not supported by bashi",
            )
            self.assertTrue(is_supported_version(name, pkv.parse(str(version))))

        for name, version in [
            (CMAKE, 2.78),
            (CMAKE, 4546.546),
            (BOOST, 0.93),
            (UBUNTU, 10.05),
            (CXX_STANDARD, 5),
        ]:
            # test if version is unsupported for the following tests
            self.assertFalse(
                pkv.parse(str(version)) in parse_to_version(VERSIONS[name]),
                f"{name} {version} is supported by bashi",
            )
            self.assertFalse(is_supported_version(name, pkv.parse(str(version))))
