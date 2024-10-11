# pylint: disable=missing-docstring
import unittest
import io

from collections import OrderedDict as OD
import packaging.version as pkv
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import VERSIONS, NvccHostSupport
from bashi.filter_compiler import compiler_filter_typechecked
from bashi.filter_backend import backend_filter_typechecked


class TestNoNvccHostCompiler(unittest.TestCase):
    def test_valid_combination_rule_c1(self):
        self.assertTrue(
            compiler_filter_typechecked(
                OD({HOST_COMPILER: ppv((GCC, 10)), DEVICE_COMPILER: ppv((NVCC, 11.2))})
            )
        )

        # version should not matter
        self.assertTrue(
            compiler_filter_typechecked(
                OD({HOST_COMPILER: ppv((CLANG, 0)), DEVICE_COMPILER: ppv((NVCC, 0))})
            )
        )

        self.assertTrue(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG, 0)),
                        DEVICE_COMPILER: ppv((NVCC, 0)),
                        CMAKE: ppv((CMAKE, "3.23")),
                        BOOST: ppv((BOOST, "1.81")),
                    }
                )
            )
        )

        # if HOST_COMPILER does not exist in the row, it should pass because HOST_COMPILER can be
        # added at the next round
        self.assertTrue(
            compiler_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 0)),
                        CMAKE: ppv((CMAKE, "3.23")),
                        BOOST: ppv((BOOST, "1.81")),
                    }
                )
            )
        )

    def test_invalid_combination_rule_c1(self):
        reason_msg1 = io.StringIO()
        self.assertFalse(
            compiler_filter_typechecked(
                OD({HOST_COMPILER: ppv((NVCC, 11.2)), DEVICE_COMPILER: ppv((NVCC, 11.2))}),
                reason_msg1,
            )
        )
        self.assertEqual(reason_msg1.getvalue(), "nvcc is not allowed as host compiler")

        reason_msg2 = io.StringIO()
        self.assertFalse(
            compiler_filter_typechecked(
                OD({HOST_COMPILER: ppv((NVCC, 11.2)), DEVICE_COMPILER: ppv((GCC, 11))}), reason_msg2
            )
        )
        self.assertEqual(reason_msg2.getvalue(), "nvcc is not allowed as host compiler")

        reason_msg3 = io.StringIO()
        self.assertFalse(
            compiler_filter_typechecked(
                OD({HOST_COMPILER: ppv((NVCC, 12.2)), DEVICE_COMPILER: ppv((HIPCC, 5.1))}),
                reason_msg3,
            )
        )
        self.assertEqual(reason_msg3.getvalue(), "nvcc is not allowed as host compiler")

        reason_msg4 = io.StringIO()
        self.assertFalse(
            compiler_filter_typechecked(OD({HOST_COMPILER: ppv((NVCC, 10.2))}), reason_msg4)
        )
        self.assertEqual(reason_msg4.getvalue(), "nvcc is not allowed as host compiler")


class TestSupportedNvccHostCompiler(unittest.TestCase):
    def test_invalid_combination_rule_c2(self):
        for compiler_name in [CLANG_CUDA, HIPCC, ICPX, NVCC]:
            for compiler_version in ["0", "13", "32a2"]:
                reason_msg = io.StringIO()
                self.assertFalse(
                    compiler_filter_typechecked(
                        OD(
                            {
                                HOST_COMPILER: ppv((compiler_name, compiler_version)),
                                DEVICE_COMPILER: ppv((NVCC, "12.3")),
                            }
                        ),
                        reason_msg,
                    )
                )
                # NVCC is filtered by rule n1
                if compiler_name != NVCC:
                    self.assertEqual(
                        reason_msg.getvalue(),
                        "only gcc and clang are allowed as nvcc host compiler",
                    )

        reason_msg1 = io.StringIO()
        self.assertFalse(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((HIPCC, "5.3")),
                        DEVICE_COMPILER: ppv((NVCC, "12.3")),
                        CMAKE: ppv((CMAKE, "3.18")),
                        BOOST: ppv((BOOST, "1.81.0")),
                    }
                ),
                reason_msg1,
            )
        )
        self.assertEqual(
            reason_msg1.getvalue(),
            "only gcc and clang are allowed as nvcc host compiler",
        )

        reason_msg2 = io.StringIO()
        self.assertFalse(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((HIPCC, "5.3")),
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: ppv(
                            (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON)
                        ),
                        DEVICE_COMPILER: ppv((NVCC, "12.3")),
                    }
                ),
                reason_msg2,
            )
        )
        self.assertEqual(
            reason_msg2.getvalue(),
            "only gcc and clang are allowed as nvcc host compiler",
        )

    def test_valid_combination_rule_c2(self):
        for compiler_name in [GCC, CLANG]:
            for compiler_version in ["0", "7", "10"]:
                self.assertTrue(
                    compiler_filter_typechecked(
                        OD(
                            {
                                HOST_COMPILER: ppv((compiler_name, compiler_version)),
                                DEVICE_COMPILER: ppv((NVCC, "12.3")),
                            }
                        )
                    )
                )

        self.assertTrue(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, "10")),
                        DEVICE_COMPILER: ppv((NVCC, "11.5")),
                        BOOST: ppv((BOOST, "1.84.0")),
                        CMAKE: ppv((CMAKE, "3.23")),
                    }
                )
            )
        )
        self.assertTrue(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG, "7")),
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: ppv(
                            (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON)
                        ),
                        DEVICE_COMPILER: ppv((NVCC, "10.1")),
                    }
                )
            )
        )


class TestNvccHostSupportClass(unittest.TestCase):
    def test_sorting(self):
        data1: List[NvccHostSupport] = [
            NvccHostSupport("11.8", "12"),
            NvccHostSupport("10.1", "10"),
            NvccHostSupport("12.3", "13"),
            NvccHostSupport("11.2", "14"),
            NvccHostSupport("11.4", "10"),
            NvccHostSupport("11.5", "12"),
            NvccHostSupport("11.3", "9"),
        ]

        expect_data1: List[NvccHostSupport] = [
            NvccHostSupport("10.1", "10"),
            NvccHostSupport("11.2", "14"),
            NvccHostSupport("11.3", "9"),
            NvccHostSupport("11.4", "10"),
            NvccHostSupport("11.5", "12"),
            NvccHostSupport("11.8", "12"),
            NvccHostSupport("12.3", "13"),
        ]

        data1.sort()

        self.assertEqual(
            data1,
            expect_data1,
            f"\ngiven   : {[str(x) for x in data1]}\n"
            f"expected: {[str(x) for x in expect_data1]}",
        )

        # the comparision operator does not respect the host version
        data2: List[NvccHostSupport] = [
            NvccHostSupport("11.8", "12"),
            NvccHostSupport("12.3", "13"),
            NvccHostSupport("12.3", "12"),
            NvccHostSupport("11.2", "14"),
        ]

        expect_data2: List[NvccHostSupport] = [
            NvccHostSupport("11.2", "14"),
            NvccHostSupport("11.8", "12"),
            NvccHostSupport("12.3", "13"),
            NvccHostSupport("12.3", "12"),
        ]

        data2.sort()

        self.assertEqual(
            data2,
            expect_data2,
            f"\ngiven   : {[str(x) for x in data2]}\n"
            f"expected: {[str(x) for x in expect_data2]}",
        )

        data3: List[NvccHostSupport] = [
            NvccHostSupport("11.8", "12"),
            NvccHostSupport("10.1", "10"),
            NvccHostSupport("12.3", "13"),
            NvccHostSupport("11.2", "14"),
            NvccHostSupport("11.4", "10"),
            NvccHostSupport("11.5", "12"),
            NvccHostSupport("11.3", "9"),
        ]

        expect_data3: List[NvccHostSupport] = [
            NvccHostSupport("12.3", "13"),
            NvccHostSupport("11.8", "12"),
            NvccHostSupport("11.5", "12"),
            NvccHostSupport("11.4", "10"),
            NvccHostSupport("11.3", "9"),
            NvccHostSupport("11.2", "14"),
            NvccHostSupport("10.1", "10"),
        ]

        data3.sort(reverse=True)

        self.assertEqual(
            data3,
            expect_data3,
            f"\ngiven   : {[str(x) for x in data3]}\n"
            f"expected: {[str(x) for x in expect_data3]}",
        )


class TestNvccSupportedGccVersion(unittest.TestCase):
    def test_valid_combination_general_algorithm_rule_c5(self):
        # this tests checks, if also version are respected, which are located between two defined
        # nvcc versions
        # e.g the following was defined:
        # [ ..., NvccHostSupport("12.0", "12"), NvccHostSupport("11.4", "11"), ...]
        # nvcc 11.5 should also pass with gcc 11 and not pass with gcc 12
        expected_results = [
            ("10.0", "6", True),
            ("10.0", "7", True),
            ("10.0", "8", False),
            ("10.0", "13", False),
            ("10.1", "6", True),
            ("10.1", "7", True),
            ("10.1", "8", True),
            ("10.1", "13", False),
            ("10.2", "6", True),
            ("10.2", "7", True),
            ("10.2", "8", True),
            ("10.2", "13", False),
            ("11.0", "6", True),
            ("11.0", "9", True),
            ("11.0", "10", False),
            ("11.0", "12", False),
            ("11.3", "8", True),
            ("11.3", "9", True),
            ("11.3", "10", True),
            ("11.3", "11", False),
            ("11.3", "12", False),
            ("11.4", "8", True),
            ("11.4", "9", True),
            ("11.4", "10", True),
            ("11.4", "11", True),
            ("11.4", "12", False),
            ("11.5", "8", True),
            ("11.5", "9", True),
            ("11.5", "10", True),
            ("11.5", "11", True),
            ("11.5", "12", False),
            ("11.6", "8", True),
            ("11.6", "9", True),
            ("11.6", "10", True),
            ("11.6", "11", True),
            ("11.6", "12", False),
        ]

        for nvcc_version, gcc_version, expected_filter_return_value in expected_results:
            reason_msg = io.StringIO()
            self.assertEqual(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((GCC, gcc_version)),
                            DEVICE_COMPILER: ppv((NVCC, nvcc_version)),
                        }
                    ),
                    reason_msg,
                ),
                expected_filter_return_value,
                f"the filter for the combination of nvcc {nvcc_version} + gcc {gcc_version} "
                f"should return {expected_filter_return_value}",
            )
            if not expected_filter_return_value:
                self.assertEqual(
                    reason_msg.getvalue(),
                    f"nvcc {nvcc_version} " f"does not support gcc {gcc_version}",
                )

    def test_valid_combination_max_gcc_rule_c5(self):
        # change the version, if you added a new cuda release
        # this test is a guard to be sure, that the following test contains the latest nvcc release
        latest_covered_nvcc_release = "12.3"
        self.assertEqual(
            latest_covered_nvcc_release,
            str(VERSIONS[NVCC][-1]),
            f"The tests cases covers up to nvcc version {latest_covered_nvcc_release}.\n"
            f"VERSION[NVCC] defines nvcc {VERSIONS[NVCC][-1]} as latest supported version.",
        )

        # add the latest supported gcc version for a supported nvcc version and also the successor
        # gcc version
        expected_results = [
            ("10.0", "7", True),
            ("10.0", "8", False),
            ("10.1", "8", True),
            ("10.1", "9", False),
            ("10.2", "8", True),
            ("10.2", "9", False),
            ("11.0", "9", True),
            ("11.0", "10", False),
            ("11.1", "10", True),
            ("11.1", "11", False),
            ("11.2", "10", True),
            ("11.2", "11", False),
            ("11.3", "10", True),
            ("11.3", "11", False),
            ("11.4", "11", True),
            ("11.4", "12", False),
            ("11.5", "11", True),
            ("11.5", "12", False),
            ("11.6", "11", True),
            ("11.6", "12", False),
            ("11.7", "11", True),
            ("11.7", "12", False),
            ("11.8", "11", True),
            ("11.8", "12", False),
            ("12.0", "12", True),
            ("12.0", "13", False),
            ("12.1", "12", True),
            ("12.1", "13", False),
            ("12.2", "12", True),
            ("12.2", "13", False),
            ("12.3", "12", True),
            ("12.3", "13", False),
        ]

        for nvcc_version, gcc_version, expected_filter_return_value in expected_results:
            reason_msg = io.StringIO()
            self.assertEqual(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((GCC, gcc_version)),
                            DEVICE_COMPILER: ppv((NVCC, nvcc_version)),
                        }
                    ),
                    reason_msg,
                ),
                expected_filter_return_value,
                f"the filter for the combination of nvcc {nvcc_version} + gcc {gcc_version} "
                f"should return {expected_filter_return_value}",
            )
            if not expected_filter_return_value:
                self.assertEqual(
                    reason_msg.getvalue(),
                    f"nvcc {nvcc_version} " f"does not support gcc {gcc_version}",
                )

    def test_valid_multi_row_entries_gcc_rule_c5(self):
        self.assertTrue(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 10)),
                        DEVICE_COMPILER: ppv((NVCC, 11.2)),
                        CMAKE: ppv((CMAKE, 3.18)),
                        BOOST: ppv((BOOST, 1.78)),
                    }
                )
            )
        )

        self.assertTrue(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 12)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1)),
                        DEVICE_COMPILER: ppv((NVCC, 12.1)),
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: ppv(
                            (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)
                        ),
                    }
                )
            )
        )

    def test_invalid_multi_row_entries_gcc_rule_c5(self):
        reason_msg1 = io.StringIO()
        self.assertFalse(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 13)),
                        DEVICE_COMPILER: ppv((NVCC, 11.2)),
                        CMAKE: ppv((CMAKE, 3.18)),
                        BOOST: ppv((BOOST, 1.78)),
                    }
                ),
                reason_msg1,
            ),
        )
        self.assertEqual(
            reason_msg1.getvalue(),
            "nvcc 11.2 does not support gcc 13",
        )

        reason_msg2 = io.StringIO()
        self.assertFalse(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 12)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8)),
                        DEVICE_COMPILER: ppv((NVCC, 11.8)),
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: ppv(
                            (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)
                        ),
                    }
                ),
                reason_msg2,
            )
        )
        self.assertEqual(
            reason_msg2.getvalue(),
            "nvcc 11.8 does not support gcc 12",
        )

    def test_unknown_combination_gcc_rule_c5(self):
        # test an unsupported nvcc version
        # we assume, that the nvcc supports all gcc versions
        unsupported_nvcc_version = 42.0
        self.assertFalse(
            unsupported_nvcc_version in VERSIONS[NVCC],
            f"for the test, it is required that nvcc {unsupported_nvcc_version} is an unsupported "
            "version",
        )

        self.assertTrue(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 12)),
                        DEVICE_COMPILER: ppv((NVCC, unsupported_nvcc_version)),
                    }
                ),
            ),
            f"nvcc {unsupported_nvcc_version} should pass the filter, because it is unknown "
            "version",
        )


class TestNvccSupportedClangVersion(unittest.TestCase):
    def test_valid_combination_max_clang_rule_c6_rule_c7(self):
        # change the version, if you added a new cuda release
        # this test is a guard to be sure, that the following test contains the latest nvcc release
        latest_covered_nvcc_release = "12.3"
        self.assertEqual(
            latest_covered_nvcc_release,
            str(VERSIONS[NVCC][-1]),
            f"The tests cases covers up to nvcc version {latest_covered_nvcc_release}.\n"
            f"VERSION[NVCC] defines nvcc {VERSIONS[NVCC][-1]} as latest supported version.",
        )

        # add the latest supported clang version for a supported nvcc version and also the successor
        # clang version
        expected_results = [
            ("10.0", "6", True),
            ("10.0", "7", False),
            ("10.1", "8", True),
            ("10.1", "9", False),
            ("10.2", "8", True),
            ("10.2", "9", False),
            ("11.0", "9", True),
            ("11.0", "10", False),
            ("11.1", "10", True),
            ("11.1", "11", False),
            ("11.2", "11", True),
            ("11.2", "12", False),
            # Rule: v4
            # because of compiler bugs, clang is disabled for CUDA 11.3 until 11.5
            ("11.3", "11", False),
            ("11.3", "12", False),
            ("11.4", "12", False),
            ("11.4", "13", False),
            ("11.5", "12", False),
            ("11.5", "13", False),
            ("11.6", "13", True),
            ("11.6", "14", False),
            ("11.7", "13", True),
            ("11.7", "14", False),
            ("11.8", "13", True),
            ("11.8", "14", False),
            ("12.0", "14", True),
            ("12.0", "15", False),
            ("12.1", "15", True),
            ("12.1", "16", False),
            ("12.2", "15", True),
            ("12.2", "16", False),
            ("12.3", "16", True),
            ("12.3", "17", False),
        ]

        for nvcc_version, clang_version, expected_filter_return_value in expected_results:
            reason_msg = io.StringIO()
            self.assertEqual(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG, clang_version)),
                            DEVICE_COMPILER: ppv((NVCC, nvcc_version)),
                        }
                    ),
                    reason_msg,
                ),
                expected_filter_return_value,
                f"the filter for the combination of nvcc {nvcc_version} + clang {clang_version} "
                f"should return {expected_filter_return_value}",
            )

            if not expected_filter_return_value:
                # for nvcc version 11.3 to 11.5, rule v4 is used
                if pkv.parse(nvcc_version) >= pkv.parse("11.3") and pkv.parse(
                    nvcc_version
                ) <= pkv.parse("11.5"):
                    self.assertEqual(
                        reason_msg.getvalue(),
                        "clang as host compiler is disabled for nvcc 11.3 to 11.5",
                    )
                else:
                    self.assertEqual(
                        reason_msg.getvalue(),
                        f"nvcc {nvcc_version} " f"does not support clang {clang_version}",
                    )

    def test_valid_multi_row_entries_clang_rule_c6(self):
        self.assertTrue(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG, 10)),
                        DEVICE_COMPILER: ppv((NVCC, 11.2)),
                        CMAKE: ppv((CMAKE, 3.18)),
                        BOOST: ppv((BOOST, 1.78)),
                    }
                )
            )
        )

        self.assertTrue(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG, 12)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1)),
                        DEVICE_COMPILER: ppv((NVCC, 12.1)),
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: ppv(
                            (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)
                        ),
                    }
                )
            )
        )

    def test_invalid_multi_row_entries_clang_rule_c6(self):
        reason_msg1 = io.StringIO()
        self.assertFalse(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG, 13)),
                        DEVICE_COMPILER: ppv((NVCC, 11.2)),
                        CMAKE: ppv((CMAKE, 3.18)),
                        BOOST: ppv((BOOST, 1.78)),
                    }
                ),
                reason_msg1,
            ),
        )
        self.assertEqual(
            reason_msg1.getvalue(),
            "nvcc 11.2 does not support clang 13",
        )

        reason_msg2 = io.StringIO()
        self.assertFalse(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG, 16)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8)),
                        DEVICE_COMPILER: ppv((NVCC, 11.8)),
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE: ppv(
                            (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)
                        ),
                    }
                ),
                reason_msg2,
            )
        )
        self.assertEqual(
            reason_msg2.getvalue(),
            "nvcc 11.8 does not support clang 16",
        )

    def test_unknown_combination_clang_rule_c6(self):
        # test an unsupported nvcc version
        # we assume, that the nvcc supports all gcc versions
        unsupported_nvcc_version = 42.0
        self.assertFalse(
            unsupported_nvcc_version in VERSIONS[NVCC],
            f"for the test, it is required that nvcc {unsupported_nvcc_version} is an unsupported "
            "version",
        )

        self.assertTrue(
            compiler_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG, 12)),
                        DEVICE_COMPILER: ppv((NVCC, unsupported_nvcc_version)),
                    }
                ),
            ),
            f"nvcc {unsupported_nvcc_version} should pass the filter, because it is unknown "
            "version",
        )

    def test_no_clang_as_host_compiler_rule_c7(self):
        for nvcc_version in [11.3, 11.4, 11.5]:
            for clang_version in [0, 7, 56]:
                reason_msg = io.StringIO()
                self.assertFalse(
                    compiler_filter_typechecked(
                        OD(
                            {
                                HOST_COMPILER: ppv((CLANG, clang_version)),
                                DEVICE_COMPILER: ppv((NVCC, nvcc_version)),
                            }
                        ),
                        reason_msg,
                    ),
                    f"nvcc {nvcc_version} does not allow clang as host compiler",
                )
                self.assertEqual(
                    reason_msg.getvalue(),
                    "clang as host compiler is disabled for nvcc 11.3 to 11.5",
                )

    def test_no_clang_as_host_compiler_multi_row_rule_c7(self):
        for nvcc_version in [11.3, 11.4, 11.5]:
            for clang_version in [0, 10, 78]:
                reason_msg = io.StringIO()
                self.assertFalse(
                    compiler_filter_typechecked(
                        OD(
                            {
                                HOST_COMPILER: ppv((CLANG, clang_version)),
                                ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                    (ALPAKA_ACC_GPU_CUDA_ENABLE, nvcc_version)
                                ),
                                DEVICE_COMPILER: ppv((NVCC, nvcc_version)),
                                CMAKE: ppv((CMAKE, 3.18)),
                            }
                        ),
                        reason_msg,
                    ),
                    f"nvcc {nvcc_version} does not allow clang as host compiler",
                )
                self.assertEqual(
                    reason_msg.getvalue(),
                    "clang as host compiler is disabled for nvcc 11.3 to 11.5",
                )


class TestNvccCompilerFilter(unittest.TestCase):
    def test_nvcc_requires_sane_cuda_backend_version_pass_c15(self):
        for version in ("10.1", "11.2", "12.3"):
            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((NVCC, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, version)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((GCC, 7)),
                            DEVICE_COMPILER: ppv((NVCC, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, version)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, version)),
                            HOST_COMPILER: ppv((GCC, 7)),
                            DEVICE_COMPILER: ppv((NVCC, version)),
                        }
                    )
                )
            )

            self.assertTrue(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((CLANG, 6)),
                            DEVICE_COMPILER: ppv((NVCC, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, version)),
                        }
                    )
                )
            )

    def test_nvcc_requires_sane_cuda_backend_version_not_pass_c15(self):
        for version in ("10.1", "11.2", "12.3"):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((NVCC, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    ),
                    reason_msg1,
                )
            )
            self.assertEqual(
                reason_msg1.getvalue(), "nvcc and CUDA backend needs to have the same version"
            )

            reason_msg2 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((NVCC, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, "11.8")),
                        }
                    ),
                    reason_msg2,
                )
            )
            self.assertEqual(
                reason_msg2.getvalue(), "nvcc and CUDA backend needs to have the same version"
            )

            reason_msg3 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((GCC, 7)),
                            DEVICE_COMPILER: ppv((NVCC, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    ),
                    reason_msg3,
                )
            )
            self.assertEqual(
                reason_msg3.getvalue(), "nvcc and CUDA backend needs to have the same version"
            )

            reason_msg4 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1)),
                            HOST_COMPILER: ppv((GCC, 7)),
                            DEVICE_COMPILER: ppv((NVCC, version)),
                        }
                    ),
                    reason_msg4,
                )
            )
            self.assertEqual(
                reason_msg4.getvalue(), "nvcc and CUDA backend needs to have the same version"
            )

            reason_msg5 = io.StringIO()
            self.assertFalse(
                compiler_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((GCC, 7)),
                            DEVICE_COMPILER: ppv((NVCC, version)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0)),
                        }
                    ),
                    reason_msg5,
                )
            )
            self.assertEqual(
                reason_msg5.getvalue(), "nvcc and CUDA backend needs to have the same version"
            )

    def test_nvcc_requires_disabled_hip_backend_c16(self):
        self.assertTrue(
            compiler_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 11.2)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                    }
                ),
            )
        )

        reason_msg1 = io.StringIO()
        self.assertFalse(
            compiler_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 11.3)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                    }
                ),
                reason_msg1,
            )
        )
        self.assertEqual(reason_msg1.getvalue(), "nvcc does not support the HIP backend.")

    def test_nvcc_requires_disabled_sycl_backend_c17(self):
        self.assertTrue(
            compiler_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 11.2)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                    }
                ),
            )
        )

        reason_msg1 = io.StringIO()
        self.assertFalse(
            compiler_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 11.3)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                    }
                ),
                reason_msg1,
            )
        )
        self.assertEqual(reason_msg1.getvalue(), "nvcc does not support the SYCL backend.")

    def test_disallow_disabled_cuda_backend_for_nvcc_device_compiler_b7(self):
        reason_msg1 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 12.2)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                    }
                ),
                reason_msg1,
            )
        )
        self.assertEqual(reason_msg1.getvalue(), "CUDA backend needs to be enabled for nvcc")

        reason_msg2 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, 9)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        DEVICE_COMPILER: ppv((NVCC, 12.2)),
                    }
                ),
                reason_msg2,
            )
        )
        self.assertEqual(reason_msg2.getvalue(), "CUDA backend needs to be enabled for nvcc")

    def test_check_if_cuda_backend_is_disabled_for_no_cuda_compiler_pass_b7(self):
        for compiler_name in set(COMPILERS) - set([NVCC, CLANG_CUDA]):
            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_GPU_CUDA_ENABLE should be off for {compiler_name}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_GPU_CUDA_ENABLE should be off for {compiler_name}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_GPU_CUDA_ENABLE should be off for {compiler_name}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            CMAKE: ppv((CMAKE, 3.18)),
                            HOST_COMPILER: ppv((compiler_name, 9999)),
                            DEVICE_COMPILER: ppv((compiler_name, 9999)),
                            BOOST: ppv((BOOST, "1.78.0")),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        }
                    )
                ),
                f"ALPAKA_ACC_GPU_CUDA_ENABLE should be off for {compiler_name}",
            )

    def test_check_if_cuda_backend_is_disabled_for_unsupported_host_compiler_b8(self):
        for host_compiler, cuda_sdk in (
            ((GCC, 7), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
            ((CLANG, 7), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8)),
            ((CLANG_CUDA, 16), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.0)),
            ((NVCC, 12.0), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0)),
        ):
            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((host_compiler[0], host_compiler[1])),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((cuda_sdk[0], cuda_sdk[1])),
                        }
                    )
                ),
                f"{host_compiler[0]} {host_compiler[1]} + CUDA {cuda_sdk[1]}",
            )

        for host_compiler, cuda_sdk in (
            ((HIPCC, 5.1), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
            ((ICPX, "2023.1.0"), (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8)),
        ):
            reason_msg1 = io.StringIO()

            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((host_compiler[0], host_compiler[1])),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((cuda_sdk[0], cuda_sdk[1])),
                        }
                    ),
                    reason_msg1,
                ),
                f"{host_compiler[0]} {host_compiler[1]} + CUDA {cuda_sdk[1]}",
            )
            self.assertEqual(
                reason_msg1.getvalue(),
                f"host-compiler {host_compiler[0]} does not support the CUDA backend",
            )

    def test_nvcc_and_cuda_backend_needs_same_version_b9(self):
        for version in (10.1, 11.2, 12.3):
            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((NVCC, version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, version)),
                        }
                    ),
                )
            )

        for version_nvcc, version_cuda in ((10.1, 10.2), (11.2, 10.1), (12.3, 12.2)):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((NVCC, version_nvcc)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, version_cuda)
                            ),
                        }
                    ),
                    reason_msg1,
                )
            )
            self.assertEqual(
                reason_msg1.getvalue(), "CUDA backend and nvcc needs to have the same version"
            )

    def test_gcc_host_compiler_support_cuda_sdk_b10(self):
        for gcc_version, version_cuda in ((5, 10.2), (9, 11.8), (11, 12.2)):
            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((GCC, gcc_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, version_cuda)
                            ),
                        }
                    ),
                ),
                f"gcc {gcc_version} + CUDA {version_cuda}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((GCC, gcc_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, version_cuda)
                            ),
                            DEVICE_COMPILER: ppv((NVCC, version_cuda)),
                        }
                    ),
                ),
                f"gcc {gcc_version} + CUDA {version_cuda}",
            )

        for gcc_version, version_cuda in ((12, 10.2), (15, 11.8), (13, 12.2)):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((GCC, gcc_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, version_cuda)
                            ),
                        }
                    ),
                    reason_msg1,
                ),
                f"gcc {gcc_version} + CUDA {version_cuda}",
            )
            self.assertEqual(
                reason_msg1.getvalue(), f"CUDA {version_cuda} does not support gcc {gcc_version}"
            )

    def test_no_clang_compiler_for_cuda_113_to_115_b11(self):
        for clang_version in (7, 9, 12):
            for cuda_version in (11.3, 11.4, 11.5):
                reason_msg1 = io.StringIO()
                self.assertFalse(
                    backend_filter_typechecked(
                        OD(
                            {
                                HOST_COMPILER: ppv((CLANG, clang_version)),
                                ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                    (ALPAKA_ACC_GPU_CUDA_ENABLE, cuda_version)
                                ),
                            }
                        ),
                        reason_msg1,
                    ),
                    f"clang {clang_version} + CUDA {cuda_version}",
                )
                self.assertEqual(
                    reason_msg1.getvalue(),
                    "clang as host compiler is disabled for CUDA 11.3 to 11.5",
                )

    def test_clang_host_compiler_support_cuda_sdk_b12(self):
        for clang_version, version_cuda in ((5, 10.2), (9, 11.8), (11, 12.2)):
            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG, clang_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, version_cuda)
                            ),
                        }
                    ),
                ),
                f"clang {clang_version} + CUDA {version_cuda}",
            )

            self.assertTrue(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG, clang_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, version_cuda)
                            ),
                            DEVICE_COMPILER: ppv((NVCC, version_cuda)),
                        }
                    ),
                ),
                f"clang {clang_version} + CUDA {version_cuda}",
            )

        for clang_version, version_cuda in ((12, 10.2), (15, 11.8), (17, 12.2)):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((CLANG, clang_version)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv(
                                (ALPAKA_ACC_GPU_CUDA_ENABLE, version_cuda)
                            ),
                        }
                    ),
                    reason_msg1,
                ),
                f"clang {clang_version} + CUDA {version_cuda}",
            )
            self.assertEqual(
                reason_msg1.getvalue(),
                f"CUDA {version_cuda} does not support clang {clang_version}",
            )

    def test_unsupported_cuda_device_compiler_b13(self):
        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 11.2)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                    }
                ),
            ),
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((CLANG_CUDA, 15)),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                    }
                ),
            ),
        )

        for device_compiler in set(COMPILERS) - set([NVCC, CLANG_CUDA]):
            reason_msg1 = io.StringIO()
            self.assertFalse(
                backend_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((device_compiler, 7)),
                            ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.2)),
                        }
                    ),
                    reason_msg1,
                ),
                f"device-compiler {DEVICE_COMPILER} + CUDA 11.2",
            )
            self.assertEqual(
                reason_msg1.getvalue(),
                f"{device_compiler} does not support the CUDA backend",
            )

    def test_cuda_and_hip_backend_cannot_be_active_at_the_same_time_b14(self):
        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                    }
                ),
            )
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                    }
                ),
            )
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                    }
                ),
            )
        )

        reason_msg1 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.7)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                    }
                ),
                reason_msg1,
            )
        )
        self.assertEqual(
            reason_msg1.getvalue(), "The HIP and CUDA backend cannot be enabled on the same time."
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        CMAKE: ppv((CMAKE, 3.18)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, OFF)),
                        BOOST: ppv((BOOST, "1.78.0")),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.2)),
                    }
                ),
            )
        )

        reason_msg2 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        CMAKE: ppv((CMAKE, 3.18)),
                        ALPAKA_ACC_GPU_HIP_ENABLE: ppv((ALPAKA_ACC_GPU_HIP_ENABLE, ON)),
                        BOOST: ppv((BOOST, "1.78.0")),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.3)),
                    }
                ),
                reason_msg2,
            )
        )
        self.assertEqual(
            reason_msg2.getvalue(), "The HIP and CUDA backend cannot be enabled on the same time."
        )

    def test_cuda_and_sycl_backend_cannot_be_active_at_the_same_time_b15(self):
        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                    }
                ),
            )
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                    }
                ),
            )
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                    }
                ),
            )
        )

        reason_msg1 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 11.7)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                    }
                ),
                reason_msg1,
            )
        )
        self.assertEqual(
            reason_msg1.getvalue(), "The SYCL and CUDA backend cannot be enabled on the same time."
        )

        self.assertTrue(
            backend_filter_typechecked(
                OD(
                    {
                        CMAKE: ppv((CMAKE, 3.18)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, OFF)),
                        BOOST: ppv((BOOST, "1.78.0")),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 10.2)),
                    }
                ),
            )
        )

        reason_msg2 = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        CMAKE: ppv((CMAKE, 3.18)),
                        ALPAKA_ACC_SYCL_ENABLE: ppv((ALPAKA_ACC_SYCL_ENABLE, ON)),
                        BOOST: ppv((BOOST, "1.78.0")),
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, 12.3)),
                    }
                ),
                reason_msg2,
            )
        )
        self.assertEqual(
            reason_msg2.getvalue(), "The SYCL and CUDA backend cannot be enabled on the same time."
        )


def test_valid_cuda_and_cxx_combinations_b18(self):
    self.assertTrue(
        backend_filter_typechecked(
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, "10.0")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 15)),
                }
            )
        )
    )
    self.assertTrue(
        backend_filter_typechecked(
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, "10.2")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 15)),
                }
            )
        )
    )
    self.assertTrue(
        backend_filter_typechecked(
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, "10.5")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 15)),
                }
            )
        )
    )
    self.assertTrue(
        backend_filter_typechecked(
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, "11.0")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 16)),
                }
            )
        )
    )
    self.assertTrue(
        backend_filter_typechecked(
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, "11.0")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 15)),
                }
            )
        )
    )
    self.assertTrue(
        backend_filter_typechecked(
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, "11.5")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 15)),
                }
            )
        )
    )
    self.assertTrue(
        backend_filter_typechecked(
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, "12.0")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 18)),
                }
            )
        )
    )
    self.assertTrue(
        backend_filter_typechecked(
            OD(
                {
                    ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, "12.0")),
                    CXX_STANDARD: ppv((CXX_STANDARD, 19)),
                }
            )
        )
    )


def test_invalid_cuda_and_cxx_combinations_b18(self):
    for cxx_version in [17, 20, 25]:
        reason_msg = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, "10.1")),
                        CXX_STANDARD: ppv((CXX_STANDARD, cxx_version)),
                    }
                ),
                reason_msg,
            ),
        )
        self.assertEqual(
            reason_msg.getvalue(),
            f"cuda 10.1 does not support cxx {cxx_version}",
        )
    for cxx_version in [17, 20, 25]:
        reason_msg = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, "11.0")),
                        CXX_STANDARD: ppv((CXX_STANDARD, cxx_version)),
                    }
                ),
                reason_msg,
            ),
        )
        self.assertEqual(
            reason_msg.getvalue(),
            f"cuda 11.0 does not support cxx {cxx_version}",
        )

    for cxx_version in [20, 21, 28]:
        reason_msg = io.StringIO()
        self.assertFalse(
            backend_filter_typechecked(
                OD(
                    {
                        ALPAKA_ACC_GPU_CUDA_ENABLE: ppv((ALPAKA_ACC_GPU_CUDA_ENABLE, "12.0")),
                        CXX_STANDARD: ppv((CXX_STANDARD, cxx_version)),
                    }
                ),
                reason_msg,
            ),
        )
        self.assertEqual(
            reason_msg.getvalue(),
            f"cuda 12.0 does not support cxx {cxx_version}",
        )
