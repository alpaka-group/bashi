# pylint: disable=missing-docstring
# pylint: disable=too-many-lines
import unittest
import io
from collections import OrderedDict as OD
import packaging.version as pkv
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import (
    CompilerCxxSupport,
    ClangCudaSDKSupport,
    _get_clang_cuda_cuda_sdk_cxx_support,
    MAX_CUDA_SDK_CXX_SUPPORT,
    NVCC_CXX_SUPPORT_VERSION,
    ClangBase,
    _get_clang_base_compiler_cxx_support,
)
from bashi.filter_compiler import (
    compiler_filter_typechecked,
    _get_max_supported_cxx_version_for_cuda_sdk_for_nvcc,
    _get_max_supported_cxx_version_for_cuda_sdk_for_clang_cuda,
    _get_max_supported_cxx_version_for_cuda_sdk,
)


class TestGetMaximumSupportedCXXStandardForCUDASdk(unittest.TestCase):
    def test_get_max_supported_cxx_version_for_cuda_sdk_for_nvcc(self):
        support_list = [
            CompilerCxxSupport("10.0", "14"),
            CompilerCxxSupport("11.0", "17"),
            CompilerCxxSupport("12.0", "20"),
        ]
        for sdk_version, expected_cxx in [
            (10.0, 14),
            (11.0, 17),
            (12.0, 20),
            (10.2, 14),
            (11.8, 17),
            (12.8, 20),
            (9.0, 14),
            (13.0, 20),
        ]:
            p_sdk_version = pkv.parse(str(sdk_version))
            p_expected_cxx = pkv.parse(str(expected_cxx))
            self.assertEqual(
                _get_max_supported_cxx_version_for_cuda_sdk_for_nvcc(p_sdk_version, support_list),
                p_expected_cxx,
                f"CUDA {sdk_version}",
            )

    def test_get_max_supported_cxx_version_for_cuda_sdk_for_clang_cuda(self):
        support_list = [
            CompilerCxxSupport("12.1", "23"),
            CompilerCxxSupport("11.5", "20"),
            CompilerCxxSupport("10.0", "17"),
        ]
        for sdk_version, expected_cxx in [
            (9.0, 17),
            (10.0, 17),
            (10.2, 17),
            (11.0, 17),
            (11.5, 20),
            (11.8, 20),
            (12.0, 20),
            (12.1, 23),
            (12.8, 23),
            (13.0, 23),
        ]:
            p_sdk_version = pkv.parse(str(sdk_version))
            p_expected_cxx = pkv.parse(str(expected_cxx))
            self.assertEqual(
                _get_max_supported_cxx_version_for_cuda_sdk_for_clang_cuda(
                    p_sdk_version, support_list
                ),
                p_expected_cxx,
                f"CUDA {sdk_version}",
            )

    def test__get_max_supported_cxx_version_for_cuda_sdk(self):
        nvcc_support_list = [
            CompilerCxxSupport("10.0", "14"),
            CompilerCxxSupport("11.0", "17"),
            CompilerCxxSupport("12.0", "20"),
        ]
        clang_cuda_support_list = [
            CompilerCxxSupport("12.1", "23"),
            CompilerCxxSupport("11.5", "20"),
            CompilerCxxSupport("10.0", "17"),
        ]
        for sdk_version, expected_cxx in [
            (9.0, 17),
            (10.0, 17),
            (10.2, 17),
            (11.0, 17),
            (11.5, 20),
            (11.8, 20),
            (12.0, 20),
            (12.1, 23),
            (12.8, 23),
            (13.0, 23),
        ]:
            p_sdk_version = pkv.parse(str(sdk_version))
            p_expected_cxx = pkv.parse(str(expected_cxx))
            self.assertEqual(
                _get_max_supported_cxx_version_for_cuda_sdk(
                    p_sdk_version, nvcc_support_list, clang_cuda_support_list
                ),
                p_expected_cxx,
                f"CUDA {sdk_version}",
            )

    def test_get_clang_cuda_cuda_sdk_cxx_support_1(self):
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

        clang_cxx_support_version: List[CompilerCxxSupport] = [
            CompilerCxxSupport("9", "17"),
            CompilerCxxSupport("14", "20"),
            CompilerCxxSupport("17", "23"),
        ]

        expected_list: List[CompilerCxxSupport] = [
            CompilerCxxSupport("12.1", "23"),
            CompilerCxxSupport("11.5", "20"),
            CompilerCxxSupport("10.0", "17"),
        ]
        result = _get_clang_cuda_cuda_sdk_cxx_support(
            clang_cxx_support_version, clang_cuda_max_cuda_version
        )

        # workaround for Python <= 3.11
        # SyntaxError: f-string expression part cannot include a backslash
        new_line = "\n"
        self.assertEqual(
            result,
            expected_list,
            f"\nresult: \n{new_line.join([str(x) for x in result])}"
            f"\nexpected: \n{new_line.join([str(x) for x in expected_list])}",
        )

    def test_get_clang_cuda_cuda_sdk_cxx_support_2(self):
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

        clang_cxx_support_version: List[CompilerCxxSupport] = [
            CompilerCxxSupport("4", "14"),
            CompilerCxxSupport("9", "17"),
            CompilerCxxSupport("14", "20"),
            CompilerCxxSupport("17", "23"),
            CompilerCxxSupport("19", "26"),
        ]

        expected_list: List[CompilerCxxSupport] = [
            CompilerCxxSupport("12.1", "26"),
            CompilerCxxSupport("12.1", "23"),
            CompilerCxxSupport("11.5", "20"),
            CompilerCxxSupport("10.0", "17"),
        ]
        result = _get_clang_cuda_cuda_sdk_cxx_support(
            clang_cxx_support_version, clang_cuda_max_cuda_version
        )

        # workaround for Python <= 3.11
        # SyntaxError: f-string expression part cannot include a backslash
        new_line = "\n"

        self.assertEqual(
            result,
            expected_list,
            f"\nresult: \n{new_line.join([str(x) for x in result])}"
            f"\nexpected: \n{new_line.join([str(x) for x in expected_list])}",
        )


class TestCompilerCXXSupportFilterRules(unittest.TestCase):
    def test_ignore_combination_gcc_cxx_support_c21(self):
        self.assertTrue(compiler_filter_typechecked(OD({HOST_COMPILER: ppv((GCC, 10))})))
        self.assertTrue(compiler_filter_typechecked(OD({CXX_STANDARD: ppv((CXX_STANDARD, 20))})))
        self.assertTrue(
            compiler_filter_typechecked(
                OD({CXX_STANDARD: ppv((CXX_STANDARD, 20)), CMAKE: ppv((CMAKE, 3.18))})
            )
        )

    def test_valid_in_range_gcc_cxx_support_c21(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 0)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 16)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 16)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 9)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 9)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 10)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 10)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 10)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 11)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 11)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 11)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 9999)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                }
            ),
        ]:
            self.assertTrue(compiler_filter_typechecked(row), f"{row}")

    def test_invalid_in_range_gcc_cxx_support_c21(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 9999)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((GCC, 8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 18)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((GCC, 8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((GCC, 8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((GCC, 9)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 18)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 10)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 21)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 11)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 24)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 11)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 26)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 13)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 26)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 9999)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 26)),
                },
            ),
        ]:
            if HOST_COMPILER in row:
                compiler_type = HOST_COMPILER
            elif DEVICE_COMPILER in row:
                compiler_type = DEVICE_COMPILER
            else:
                compiler_type = ""

            reason_msg = io.StringIO()

            self.assertFalse(compiler_filter_typechecked(row, reason_msg), f"{row}")
            self.assertEqual(
                reason_msg.getvalue(),
                f"{compiler_type} gcc {row[compiler_type].version} does not support "
                f"C++{row[CXX_STANDARD].version}",
            )

    def test_valid_in_range_clang_cxx_support_c22(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((CLANG, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 0)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG, 8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG, 9)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG, 13)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG, 14)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG, 14)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG, 17)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG, 9999)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                }
            ),
        ]:
            self.assertTrue(compiler_filter_typechecked(row), f"{row}")

    def test_invalid_in_range_clang_cxx_support_c22(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((CLANG, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 9999)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG, 7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG, 8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((CLANG, 13)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((CLANG, 16)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((CLANG, 9999)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 9999)),
                },
            ),
        ]:
            if HOST_COMPILER in row:
                compiler_type = HOST_COMPILER
            elif DEVICE_COMPILER in row:
                compiler_type = DEVICE_COMPILER
            else:
                compiler_type = ""

            reason_msg = io.StringIO()

            self.assertFalse(compiler_filter_typechecked(row, reason_msg), f"{row}")
            self.assertEqual(
                reason_msg.getvalue(),
                f"{compiler_type} clang {row[compiler_type].version} does not support "
                f"C++{row[CXX_STANDARD].version}",
            )

    def test_valid_in_range_nvcc_cxx_support_c23(self):
        for row in [
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 10.1)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 0)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 11.0)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 11.0)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 12.3)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 12.8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 12.0)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 99.0)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
        ]:
            self.assertTrue(compiler_filter_typechecked(row), f"{row}")

    def test_invalid_in_range_nvcc_cxx_support_c23(self):
        for row in [
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 10.2)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 9999)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 10.2)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 11.8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                },
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((NVCC, 12.9)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                },
            ),
        ]:
            reason_msg = io.StringIO()

            self.assertFalse(compiler_filter_typechecked(row, reason_msg), f"{row}")
            self.assertEqual(
                reason_msg.getvalue(),
                f"{DEVICE_COMPILER} nvcc {row[DEVICE_COMPILER].version} does not support "
                f"C++{row[CXX_STANDARD].version}",
            )

    def test_valid_combination_cxx_cuda_backend_host_compiler_c24(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 12)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 12)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((CLANG, 15)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 11)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.1)),
                }
            ),
        ]:
            reason_msg = io.StringIO()
            self.assertTrue(
                compiler_filter_typechecked(row, reason_msg), f"{reason_msg.getvalue()}\n{row}"
            )

    def test_invalid_combination_cxx_cuda_backend_host_compiler_c24(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 12)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((CLANG, 14)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.5)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((CLANG, 16)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((GCC, 11)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.2)),
                }
            ),
        ]:
            reason_msg = io.StringIO()

            self.assertFalse(compiler_filter_typechecked(row, reason_msg), f"{row}")
            self.assertEqual(
                reason_msg.getvalue(),
                f"{row[HOST_COMPILER].name} {row[HOST_COMPILER].version} + "
                f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} + "
                f"C++ {row[CXX_STANDARD].version}: "
                f"there is no Nvcc version which support this combination",
            )

    def test_valid_in_range_clang_cuda_cxx_support_c25(self):
        for row in [
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG_CUDA, 14)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG_CUDA, 14)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG_CUDA, 14)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG_CUDA, 17)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((CLANG_CUDA, 9999)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                }
            ),
        ]:
            self.assertTrue(compiler_filter_typechecked(row), f"{row}")

    def test_invalid_in_range_clang_cuda_cxx_support_c25(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((CLANG_CUDA, 14)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((CLANG_CUDA, 16)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((CLANG_CUDA, 18)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 26)),
                },
            ),
            OD(
                {
                    HOST_COMPILER: ppv((CLANG_CUDA, 9999)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 9999)),
                },
            ),
        ]:
            if HOST_COMPILER in row:
                compiler_type = HOST_COMPILER
            elif DEVICE_COMPILER in row:
                compiler_type = DEVICE_COMPILER
            else:
                compiler_type = ""

            reason_msg = io.StringIO()

            self.assertFalse(compiler_filter_typechecked(row, reason_msg), f"{row}")
            self.assertEqual(
                reason_msg.getvalue(),
                f"{compiler_type} clang-cuda {row[compiler_type].version} does not support "
                f"C++{row[CXX_STANDARD].version}",
            )

    def test_valid_cuda_sdk_max_supported_cxx_c26(self):
        for row in [
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 9)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 11)),
                }
            ),
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.1)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.2)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.5)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                }
            ),
        ]:
            self.assertTrue(compiler_filter_typechecked(row), f"{row}")

    def test_invalid_cuda_sdk_max_supported_cxx_c26(self):
        for row in [
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.1)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                }
            ),
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 9999.0)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 99)),
                }
            ),
        ]:
            reason_msg = io.StringIO()

            self.assertFalse(compiler_filter_typechecked(row, reason_msg), f"{row}")
            self.assertEqual(
                reason_msg.getvalue(),
                f"There is not Nvcc or Clang-CUDA version which supports "
                f"C++-{row[CXX_STANDARD].version} + CUDA "
                f"{row[ALPAKA_ACC_GPU_CUDA_ENABLE].version}",
            )

    def test_cuda_sdk_cxx_support_of_clang_cuda_c27(self):
        max_clang_cuda_support_cxx = sorted(MAX_CUDA_SDK_CXX_SUPPORT, reverse=True)[0]
        max_nvcc_supported_cxx = sorted(NVCC_CXX_SUPPORT_VERSION, reverse=True)[0]

        # The test only works, if this true. If this assumption does not match the real world use
        # case anymore, the test needs to be modified.
        self.assertGreater(max_clang_cuda_support_cxx.cxx, max_nvcc_supported_cxx.cxx)

        valid_row = OD(
            {
                ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                    (ALPAKA_ACC_GPU_CUDA_ENABLE, str(max_clang_cuda_support_cxx.compiler))
                ),
                CXX_STANDARD: ppv((CXX_STANDARD, str(max_clang_cuda_support_cxx.cxx))),
            }
        )
        self.assertTrue(compiler_filter_typechecked(valid_row), f"{valid_row}")

        invalid_cuda_sdk_version = float(str(max_clang_cuda_support_cxx.compiler)) + 0.1
        invalid_row = OD(
            {
                ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                    (
                        ALPAKA_ACC_GPU_CUDA_ENABLE,
                        invalid_cuda_sdk_version,
                    )
                ),
                CXX_STANDARD: ppv((CXX_STANDARD, str(max_clang_cuda_support_cxx.cxx))),
            }
        )
        reason_msg = io.StringIO()
        self.assertFalse(compiler_filter_typechecked(invalid_row, reason_msg), f"{invalid_row}")
        self.assertEqual(
            reason_msg.getvalue(),
            f"For the potential combination of C++-{max_clang_cuda_support_cxx.cxx} + "
            f"CUDA {invalid_cuda_sdk_version} there is no "
            f"Clang-CUDA compiler which support this.",
        )


class TestClangBasedCompilerCXXSupport(unittest.TestCase):
    def test_get_clang_base_compiler_cxx_support(self):
        clang_cxx_support_version: List[CompilerCxxSupport] = [
            CompilerCxxSupport("9", "17"),
            CompilerCxxSupport("14", "20"),
            CompilerCxxSupport("17", "23"),
        ]

        given: List[ClangBase] = [
            ClangBase("2017.1", "7"),
            ClangBase("2018.1", "9"),
            ClangBase("2019.3", "10"),
            ClangBase("2020.0", "14"),
            ClangBase("2021.0", "15"),
            ClangBase("2025.0", "19"),
        ]
        expected: List[CompilerCxxSupport] = sorted(
            [
                CompilerCxxSupport("2017.1", "17"),
                CompilerCxxSupport("2018.1", "17"),
                CompilerCxxSupport("2019.3", "17"),
                CompilerCxxSupport("2020.0", "20"),
                CompilerCxxSupport("2021.0", "20"),
                CompilerCxxSupport("2025.0", "23"),
            ]
        )

        result = sorted(_get_clang_base_compiler_cxx_support(given, clang_cxx_support_version))

        # workaround for Python <= 3.11
        # SyntaxError: f-string expression part cannot include a backslash
        new_line = "\n"
        self.assertEqual(
            result,
            expected,
            f"\nresult: \n{new_line.join([str(x) for x in result])}"
            f"\nexpected: \n{new_line.join([str(x) for x in expected])}",
        )

    def test_valid_icpx_supported_cxx_c27(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((ICPX, "2025.0")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((ICPX, "2025.0")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((ICPX, "2025.0")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((ICPX, "2025.0")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                }
            ),
        ]:
            self.assertTrue(compiler_filter_typechecked(row), f"{row}")

    def test_invalid_icpx_supported_cxx_c27(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((ICPX, "2025.0")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 26)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((ICPX, "2025.0")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 99)),
                }
            ),
        ]:
            if HOST_COMPILER in row:
                compiler_type = HOST_COMPILER
            elif DEVICE_COMPILER in row:
                compiler_type = DEVICE_COMPILER
            else:
                compiler_type = ""

            reason_msg = io.StringIO()

            self.assertFalse(compiler_filter_typechecked(row, reason_msg), f"{row}")
            self.assertEqual(
                reason_msg.getvalue(),
                f"{compiler_type} icpx {row[compiler_type].version} does not support "
                f"C++{row[CXX_STANDARD].version}",
            )

    def test_valid_hipcc_supported_cxx_c28(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((HIPCC, 5.7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 14)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((HIPCC, 6.0)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 17)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((HIPCC, 6.1)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 20)),
                }
            ),
            OD(
                {
                    HOST_COMPILER: ppv((HIPCC, 5.7)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 23)),
                }
            ),
        ]:
            self.assertTrue(compiler_filter_typechecked(row), f"{row}")

    def test_invalid_hipcc_supported_cxx_c28(self):
        for row in [
            OD(
                {
                    HOST_COMPILER: ppv((HIPCC, 6.3)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 26)),
                }
            ),
            OD(
                {
                    DEVICE_COMPILER: ppv((HIPCC, 6.8)),
                    CXX_STANDARD: ppv((CXX_STANDARD, 99)),
                }
            ),
        ]:
            if HOST_COMPILER in row:
                compiler_type = HOST_COMPILER
            elif DEVICE_COMPILER in row:
                compiler_type = DEVICE_COMPILER
            else:
                compiler_type = ""

            reason_msg = io.StringIO()

            self.assertFalse(compiler_filter_typechecked(row, reason_msg), f"{row}")
            self.assertEqual(
                reason_msg.getvalue(),
                f"{compiler_type} hipcc {row[compiler_type].version} does not support "
                f"C++{row[CXX_STANDARD].version}",
            )
