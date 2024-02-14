# pylint: disable=missing-docstring
import unittest
import io

from collections import OrderedDict as OD
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import VERSIONS, NvccHostSupport
from bashi.filter_compiler_name import compiler_name_filter_typechecked
from bashi.filter_compiler_version import compiler_version_filter_typechecked


class TestNoNvccHostCompiler(unittest.TestCase):
    def test_valid_combination_rule_n1(self):
        self.assertTrue(
            compiler_name_filter_typechecked(
                OD({HOST_COMPILER: ppv((GCC, 10)), DEVICE_COMPILER: ppv((NVCC, 11.2))})
            )
        )

        # version should not matter
        self.assertTrue(
            compiler_name_filter_typechecked(
                OD({HOST_COMPILER: ppv((CLANG, 0)), DEVICE_COMPILER: ppv((NVCC, 0))})
            )
        )

        self.assertTrue(
            compiler_name_filter_typechecked(
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
            compiler_name_filter_typechecked(
                OD(
                    {
                        DEVICE_COMPILER: ppv((NVCC, 0)),
                        CMAKE: ppv((CMAKE, "3.23")),
                        BOOST: ppv((BOOST, "1.81")),
                    }
                )
            )
        )

    def test_invalid_combination_rule_n1(self):
        reason_msg1 = io.StringIO()
        self.assertFalse(
            compiler_name_filter_typechecked(
                OD({HOST_COMPILER: ppv((NVCC, 11.2)), DEVICE_COMPILER: ppv((NVCC, 11.2))}),
                reason_msg1,
            )
        )
        self.assertEqual(reason_msg1.getvalue(), "nvcc is not allowed as host compiler")

        reason_msg2 = io.StringIO()
        self.assertFalse(
            compiler_name_filter_typechecked(
                OD({HOST_COMPILER: ppv((NVCC, 11.2)), DEVICE_COMPILER: ppv((GCC, 11))}), reason_msg2
            )
        )
        self.assertEqual(reason_msg2.getvalue(), "nvcc is not allowed as host compiler")

        reason_msg3 = io.StringIO()
        self.assertFalse(
            compiler_name_filter_typechecked(
                OD({HOST_COMPILER: ppv((NVCC, 12.2)), DEVICE_COMPILER: ppv((HIPCC, 5.1))}),
                reason_msg3,
            )
        )
        self.assertEqual(reason_msg3.getvalue(), "nvcc is not allowed as host compiler")

        reason_msg4 = io.StringIO()
        self.assertFalse(
            compiler_name_filter_typechecked(OD({HOST_COMPILER: ppv((NVCC, 10.2))}), reason_msg4)
        )
        self.assertEqual(reason_msg4.getvalue(), "nvcc is not allowed as host compiler")


class TestSupportedNvccHostCompiler(unittest.TestCase):
    def test_invalid_combination_rule_n2(self):
        for compiler_name in [CLANG_CUDA, HIPCC, ICPX, NVCC]:
            for compiler_version in ["0", "13", "32a2"]:
                reason_msg = io.StringIO()
                self.assertFalse(
                    compiler_name_filter_typechecked(
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
            compiler_name_filter_typechecked(
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
            compiler_name_filter_typechecked(
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

    def test_valid_combination_rule_n2(self):
        for compiler_name in [GCC, CLANG]:
            for compiler_version in ["0", "13", "7b2"]:
                self.assertTrue(
                    compiler_name_filter_typechecked(
                        OD(
                            {
                                HOST_COMPILER: ppv((compiler_name, compiler_version)),
                                DEVICE_COMPILER: ppv((NVCC, "12.3")),
                            }
                        )
                    )
                )

        self.assertTrue(
            compiler_name_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((GCC, "13")),
                        DEVICE_COMPILER: ppv((NVCC, "11.5")),
                        BOOST: ppv((BOOST, "1.84.0")),
                        CMAKE: ppv((CMAKE, "3.23")),
                    }
                )
            )
        )
        self.assertTrue(
            compiler_name_filter_typechecked(
                OD(
                    {
                        HOST_COMPILER: ppv((CLANG, "14")),
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
    def test_valid_combination_general_algorithm_rule_v2(self):
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
                compiler_version_filter_typechecked(
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

    def test_valid_combination_max_gcc_rule_v2(self):
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
                compiler_version_filter_typechecked(
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

    def test_valid_multi_row_entries_gcc_rule_v2(self):
        self.assertTrue(
            compiler_version_filter_typechecked(
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
            compiler_version_filter_typechecked(
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

    def test_invalid_multi_row_entries_gcc_rule_v2(self):
        reason_msg1 = io.StringIO()
        self.assertFalse(
            compiler_version_filter_typechecked(
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
            compiler_version_filter_typechecked(
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

    def test_unknown_combination_gcc_rule_v2(self):
        # test an unsupported nvcc version
        # we assume, that the nvcc supports all gcc versions
        unsupported_nvcc_version = 42.0
        self.assertFalse(
            unsupported_nvcc_version in VERSIONS[NVCC],
            f"for the test, it is required that nvcc {unsupported_nvcc_version} is an unsupported "
            "version",
        )

        self.assertTrue(
            compiler_version_filter_typechecked(
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
    def test_valid_combination_max_clang_rule_v3(self):
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
            # because of compiler bugs, clang is disabled for CUDA 11.3 until 11.5
            # ("11.3", "11", False),
            # ("11.3", "12", False),
            # ("11.4", "12", False),
            # ("11.4", "13", False),
            # ("11.5", "12", False),
            # ("11.5", "13", False),
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
                compiler_version_filter_typechecked(
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
                self.assertEqual(
                    reason_msg.getvalue(),
                    f"nvcc {nvcc_version} " f"does not support clang {clang_version}",
                )

    def test_valid_multi_row_entries_clang_rule_v2(self):
        self.assertTrue(
            compiler_version_filter_typechecked(
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
            compiler_version_filter_typechecked(
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

    def test_invalid_multi_row_entries_clang_rule_v2(self):
        reason_msg1 = io.StringIO()
        self.assertFalse(
            compiler_version_filter_typechecked(
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
            compiler_version_filter_typechecked(
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

    def test_unknown_combination_clang_rule_v2(self):
        # test an unsupported nvcc version
        # we assume, that the nvcc supports all gcc versions
        unsupported_nvcc_version = 42.0
        self.assertFalse(
            unsupported_nvcc_version in VERSIONS[NVCC],
            f"for the test, it is required that nvcc {unsupported_nvcc_version} is an unsupported "
            "version",
        )

        self.assertTrue(
            compiler_version_filter_typechecked(
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
