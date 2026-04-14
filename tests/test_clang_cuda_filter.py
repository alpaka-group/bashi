# pylint: disable=missing-docstring
import unittest
import io

from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_compiler import compiler_filter_typechecked
from bashi.version.relation import VersionRelation
from bashi.row import BashiRow


class TestClangCudaOldVersions(unittest.TestCase):
    def setUp(self):
        self.version_relation = VersionRelation()

    def test_valid_clang_cuda_versions_rule_c8(self):
        for clang_cuda_version in [14, 16, 18, 78]:
            self.assertTrue(
                compiler_filter_typechecked(
                    BashiRow(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                        }
                    ),
                    self.version_relation,
                )
            )

    def test_valid_clang_cuda_versions_multi_row_rule_c8(self):
        for clang_cuda_version in [14, 16, 18, 78]:
            self.assertTrue(
                compiler_filter_typechecked(
                    BashiRow(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            CMAKE: ppv((CMAKE, 3.22)),
                        }
                    ),
                    self.version_relation,
                )
            )

    def test_invalid_clang_cuda_versions_rule_c8(self):
        for clang_cuda_version in [13, 7, 1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    BashiRow(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                        }
                    ),
                    self.version_relation,
                    reason_msg,
                )
            )
            self.assertEqual(
                reason_msg.getvalue(),
                "all clang versions older than 14 are disabled as CUDA Compiler",
            )

    def test_invalid_clang_cuda_versions_multi_row_rule_c8(self):
        for clang_cuda_version in [13, 7, 1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    BashiRow(
                        {
                            HOST_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                            DEVICE_COMPILER: ppv((CLANG_CUDA, clang_cuda_version)),
                            CMAKE: ppv((CMAKE, 3.22)),
                        }
                    ),
                    self.version_relation,
                    reason_msg,
                )
            )
            self.assertEqual(
                reason_msg.getvalue(),
                "all clang versions older than 14 are disabled as CUDA Compiler",
            )
