# pylint: disable=missing-docstring

import unittest
import random
import packaging.version as pkv
from typing import List
from bashi.versions import ClangCudaSDKSupport, get_oldest_supporting_clang_version_for_cuda


class TestgetOldestSupportingClangVersionForCuda(unittest.TestCase):
    def test_get_oldest_supporting_clang_version_for_cuda(self):
        clang_cuda_max_cuda_version: List[ClangCudaSDKSupport] = [
            ClangCudaSDKSupport("7", "9.2"),
            ClangCudaSDKSupport("8", "10.0"),
            ClangCudaSDKSupport("10", "10.1"),
            ClangCudaSDKSupport("12", "11.0"),
            ClangCudaSDKSupport("13", "11.2"),
            ClangCudaSDKSupport("14", "11.5"),
            ClangCudaSDKSupport("16", "11.8"),
            ClangCudaSDKSupport("17", "12.1"),
        ]
        # shuffle list, because we should not assume a specific ordering
        random.shuffle(clang_cuda_max_cuda_version)

        for cuda_version, expected_clang_cuda in [
            ("11.0", "12"),
            ("12.0", "17"),
            ("11.7", "16"),
            ("99.9", "0"),
            ("4", "7"),
            ("10.2", "12"),
        ]:
            self.assertEqual(
                get_oldest_supporting_clang_version_for_cuda(
                    cuda_version, clang_cuda_max_cuda_version
                ),
                pkv.parse(expected_clang_cuda),
                f"\ngiven CUDA version: {cuda_version}\nexpected Clang version: {expected_clang_cuda}",
            )
