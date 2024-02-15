# pylint: disable=missing-docstring
import unittest
import io

from collections import OrderedDict as OD
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_compiler_version import compiler_version_filter_typechecked


class TestClangCudaOldVersions(unittest.TestCase):
    def test_valid_clang_cuda_versions_rule_v5(self):
        for clang_cuda_version in [14, 16, 18, 78]:
            self.assertTrue(
                compiler_version_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                        }
                    )
                )
            )

    def test_valid_clang_cuda_versions_multi_row_rule_v5(self):
        for clang_cuda_version in [14, 16, 18, 78]:
            self.assertTrue(
                compiler_version_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            CMAKE: ppv((CMAKE, 3.22)),
                        }
                    ),
                )
            )

    def test_invalid_clang_cuda_versions_rule_v5(self):
        for clang_cuda_version in [13, 7, 1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                compiler_version_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                        }
                    ),
                    reason_msg,
                )
            )
            self.assertEqual(
                reason_msg.getvalue(),
                "all clang versions older than 14 are disabled as CUDA Compiler",
            )

    def test_invalid_clang_cuda_versions_multi_row_rule_v5(self):
        for clang_cuda_version in [13, 7, 1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                compiler_version_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            CMAKE: ppv((CMAKE, 3.22)),
                        }
                    ),
                    reason_msg,
                )
            )
            self.assertEqual(
                reason_msg.getvalue(),
                "all clang versions older than 14 are disabled as CUDA Compiler",
            )
