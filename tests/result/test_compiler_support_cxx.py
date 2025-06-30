# pylint: disable=missing-docstring
# pylint: disable=too-many-lines
import unittest

# pyright: reportPrivateUsage=false
from bashi.result_modules.cxx_compiler_support import (
    _remove_unsupported_cxx_version_for_compiler,
    _remove_unsupported_cxx_versions_for_gcc,
    _remove_unsupported_cxx_versions_for_clang,
    _remove_unsupported_cxx_versions_for_nvcc,
    _remove_unsupported_cxx_versions_for_clang_cuda,
    _remove_unsupported_cxx_versions_for_cuda,
    _remove_unsupported_cxx_versions_for_icpx,
)
from bashi.types import ParameterValuePair
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import CompilerCxxSupport, MAX_CUDA_SDK_CXX_SUPPORT
from utils_test import (
    parse_expected_val_pairs2,
    default_remove_test,
)


class TestCompilerCXXSupportResultFilter(unittest.TestCase):
    def test_older_standard_than_cxx_11(self):
        compiler_support_list = [
            CompilerCxxSupport("8", "11"),
            CompilerCxxSupport("9", "14"),
        ]
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 11)),
                ((HOST_COMPILER, GCC, 9), (CXX_STANDARD, 14)),
            ]
        )
        removed_param_value_pairs: List[ParameterValuePair] = []

        with self.assertRaises(RuntimeError):
            _remove_unsupported_cxx_version_for_compiler(
                test_param_value_pairs,
                removed_param_value_pairs,
                GCC,
                compiler_support_list,
                HOST_COMPILER,
            )

    def test_remove_unsupported_cxx_versions_for_gcc(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                # not affected
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((DEVICE_COMPILER, GCC, 6), (BOOST, "1.81.1")),
                ((UBUNTU, "20.04"), (CXX_STANDARD, 17)),
                ((CXX_STANDARD, 17), (BOOST, "1.79.2")),
                # corner cases
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 23)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 20)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 23)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 23)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 26)),
                # between
                ((HOST_COMPILER, GCC, 9), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 9), (CXX_STANDARD, 19)),
                ((HOST_COMPILER, GCC, 9), (CXX_STANDARD, 20)),
                # iterate C++ version
                ((CXX_STANDARD, 17), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 18), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 19), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 20), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 21), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 22), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 23), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 24), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 25), (DEVICE_COMPILER, GCC, 11)),
                # older than minumum specified GCC version
                ((HOST_COMPILER, GCC, 6), (CXX_STANDARD, 0)),
                ((HOST_COMPILER, GCC, 6), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, GCC, 6), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 6), (CXX_STANDARD, 20)),
                # newer than maximum specified GCC version
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 23)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 9999)),
            ]
        )

        expected_results: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                # not affected
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((DEVICE_COMPILER, GCC, 6), (BOOST, "1.81.1")),
                ((UBUNTU, "20.04"), (CXX_STANDARD, 17)),
                ((CXX_STANDARD, 17), (BOOST, "1.79.2")),
                # corner cases
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, GCC, 8), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, GCC, 10), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, GCC, 11), (CXX_STANDARD, 23)),
                # between
                ((HOST_COMPILER, GCC, 9), (CXX_STANDARD, 17)),
                # iterate C++ version
                ((CXX_STANDARD, 17), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 18), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 19), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 20), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 21), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 22), (DEVICE_COMPILER, GCC, 11)),
                ((CXX_STANDARD, 23), (DEVICE_COMPILER, GCC, 11)),
                # older than minumum specified GCC version
                ((HOST_COMPILER, GCC, 6), (CXX_STANDARD, 0)),
                ((HOST_COMPILER, GCC, 6), (CXX_STANDARD, 14)),
                # newer than maximum specified GCC version
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, GCC, 9999), (CXX_STANDARD, 23)),
            ]
        )

        default_remove_test(
            _remove_unsupported_cxx_versions_for_gcc,
            test_param_value_pairs,
            expected_results,
            self,
        )

    def test_remove_unsupported_cxx_versions_for_clang(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                # corner cases
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 23)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 20)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 23)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 23)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 26)),
                # between
                ((HOST_COMPILER, CLANG, 12), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 12), (CXX_STANDARD, 19)),
                ((HOST_COMPILER, CLANG, 13), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 13), (CXX_STANDARD, 20)),
                # older than minumum specified GCC version
                ((HOST_COMPILER, CLANG, 6), (CXX_STANDARD, 0)),
                ((HOST_COMPILER, CLANG, 6), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG, 6), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 6), (CXX_STANDARD, 20)),
                # newer than maximum specified GCC version
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 23)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 9999)),
            ]
        )

        expected_results: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                # corner cases
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, CLANG, 14), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG, 17), (CXX_STANDARD, 23)),
                # between
                ((HOST_COMPILER, CLANG, 12), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 13), (CXX_STANDARD, 17)),
                # older than minumum specified GCC version
                ((HOST_COMPILER, CLANG, 6), (CXX_STANDARD, 0)),
                ((HOST_COMPILER, CLANG, 6), (CXX_STANDARD, 14)),
                # newer than maximum specified GCC version
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG, 9999), (CXX_STANDARD, 23)),
            ]
        )

        default_remove_test(
            _remove_unsupported_cxx_versions_for_clang,
            test_param_value_pairs,
            expected_results,
            self,
        )

    def test_remove_unsupported_cxx_versions_for_nvcc(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((DEVICE_COMPILER, NVCC, 10.0), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, NVCC, 10.1), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, NVCC, 10.2), (CXX_STANDARD, 20)),
                ((DEVICE_COMPILER, NVCC, 11.0), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, NVCC, 11.8), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, NVCC, 11.2), (CXX_STANDARD, 20)),
                ((DEVICE_COMPILER, NVCC, 11.8), (CXX_STANDARD, 23)),
                ((DEVICE_COMPILER, NVCC, 12.0), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, NVCC, 12.8), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, NVCC, 12.2), (CXX_STANDARD, 20)),
                ((DEVICE_COMPILER, NVCC, 12.8), (CXX_STANDARD, 23)),
                ((DEVICE_COMPILER, NVCC, 99.8), (CXX_STANDARD, 20)),
            ]
        )

        expected_results: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((DEVICE_COMPILER, NVCC, 10.0), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, NVCC, 11.0), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, NVCC, 11.8), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, NVCC, 12.0), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, NVCC, 12.8), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, NVCC, 12.2), (CXX_STANDARD, 20)),
                ((DEVICE_COMPILER, NVCC, 99.8), (CXX_STANDARD, 20)),
            ]
        )

        default_remove_test(
            _remove_unsupported_cxx_versions_for_nvcc,
            test_param_value_pairs,
            expected_results,
            self,
        )

    def test_remove_unsupported_cxx_versions_for_clang_cuda(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((DEVICE_COMPILER, NVCC, 10.0), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, CLANG_CUDA, 8), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, CLANG_CUDA, 8), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, CLANG_CUDA, 9), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, CLANG_CUDA, 9), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, CLANG_CUDA, 10), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG_CUDA, 12), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG_CUDA, 12), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG_CUDA, 12), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG_CUDA, 14), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG_CUDA, 14), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG_CUDA, 14), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG_CUDA, 16), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG_CUDA, 16), (CXX_STANDARD, 23)),
                ((DEVICE_COMPILER, CLANG_CUDA, 17), (CXX_STANDARD, 20)),
                ((DEVICE_COMPILER, CLANG_CUDA, 17), (CXX_STANDARD, 23)),
                ((DEVICE_COMPILER, CLANG_CUDA, 17), (CXX_STANDARD, 26)),
                ((DEVICE_COMPILER, CLANG_CUDA, 9999), (CXX_STANDARD, 23)),
                ((DEVICE_COMPILER, CLANG_CUDA, 9999), (CXX_STANDARD, 99)),
            ]
        )

        expected_results: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((DEVICE_COMPILER, NVCC, 10.0), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, CLANG_CUDA, 8), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, CLANG_CUDA, 10), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, CLANG_CUDA, 9), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, CLANG_CUDA, 9), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG_CUDA, 12), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG_CUDA, 12), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG_CUDA, 14), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG_CUDA, 14), (CXX_STANDARD, 17)),
                ((HOST_COMPILER, CLANG_CUDA, 14), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, CLANG_CUDA, 16), (CXX_STANDARD, 20)),
                ((DEVICE_COMPILER, CLANG_CUDA, 17), (CXX_STANDARD, 20)),
                ((DEVICE_COMPILER, CLANG_CUDA, 17), (CXX_STANDARD, 23)),
                ((DEVICE_COMPILER, CLANG_CUDA, 9999), (CXX_STANDARD, 23)),
            ]
        )

        default_remove_test(
            _remove_unsupported_cxx_versions_for_clang_cuda,
            test_param_value_pairs,
            expected_results,
            self,
        )

    def test_remove_unsupported_cxx_versions_for_cuda(self):
        lasts_supported_version = sorted(MAX_CUDA_SDK_CXX_SUPPORT, reverse=True)[0]
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (BOOST, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (CXX_STANDARD, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (CXX_STANDARD, 23)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 8.0), (CXX_STANDARD, 14)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 8.0), (CXX_STANDARD, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.0), (CXX_STANDARD, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.0), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2), (CXX_STANDARD, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.5), (CXX_STANDARD, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.5), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0), (CXX_STANDARD, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0), (CXX_STANDARD, 23)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1), (CXX_STANDARD, 23)),
                (
                    (
                        ALPAKA_ACC_GPU_CUDA_ENABLE,
                        float(str(lasts_supported_version.compiler)) + 0.1,
                    ),
                    (CXX_STANDARD, str(lasts_supported_version.cxx)),
                ),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 9999.1), (CXX_STANDARD, 99)),
            ]
        )
        expected_results: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, CLANG, 9), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (BOOST, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (CXX_STANDARD, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF), (CXX_STANDARD, 23)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 8.0), (CXX_STANDARD, 14)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.0), (CXX_STANDARD, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2), (CXX_STANDARD, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.5), (CXX_STANDARD, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.5), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0), (CXX_STANDARD, 17)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0), (CXX_STANDARD, 20)),
                ((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1), (CXX_STANDARD, 23)),
            ]
        )

        try:
            default_remove_test(
                _remove_unsupported_cxx_versions_for_cuda,
                test_param_value_pairs,
                expected_results,
                self,
            )
        except Exception as e:
            output = "MAX_CUDA_SDK_CXX_SUPPORT:\n"
            for v in MAX_CUDA_SDK_CXX_SUPPORT:
                output += f"  {str(v)}\n"
            e.add_note(output)
            raise (e)

    def test_remove_unsupported_cxx_versions_for_icpx(self):
        test_param_value_pairs: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((DEVICE_COMPILER, NVCC, 10.0), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, ICPX, "2025.0"), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, ICPX, "2025.0"), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, ICPX, "2025.0"), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, ICPX, "2025.0"), (CXX_STANDARD, 23)),
                ((DEVICE_COMPILER, ICPX, "2025.0"), (CXX_STANDARD, 26)),
            ]
        )

        expected_results: List[ParameterValuePair] = parse_expected_val_pairs2(
            [
                ((DEVICE_COMPILER, NVCC, 10.0), (CXX_STANDARD, 14)),
                ((DEVICE_COMPILER, ICPX, "2025.0"), (CXX_STANDARD, 14)),
                ((HOST_COMPILER, ICPX, "2025.0"), (CXX_STANDARD, 17)),
                ((DEVICE_COMPILER, ICPX, "2025.0"), (CXX_STANDARD, 20)),
                ((HOST_COMPILER, ICPX, "2025.0"), (CXX_STANDARD, 23)),
            ]
        )

        default_remove_test(
            _remove_unsupported_cxx_versions_for_icpx,
            test_param_value_pairs,
            expected_results,
            self,
        )
