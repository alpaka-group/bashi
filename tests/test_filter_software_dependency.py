# pylint: disable=missing-docstring
import unittest
import io

from collections import OrderedDict as OD
from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.filter_software_dependency import software_dependency_filter_typechecked


class TestOldGCCVersionInUbuntu2004(unittest.TestCase):
    def test_valid_gcc_is_in_ubuntu_2004_d1(self):
        for gcc_version in [7, 13, 99]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD({HOST_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "22.04"))}),
                )
            )

            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((GCC, gcc_version)),
                            UBUNTU: ppv((UBUNTU, "22.04")),
                        }
                    ),
                )
            )
        for gcc_version in [7, 13, 99]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD({HOST_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "20.04"))}),
                )
            )

            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((GCC, gcc_version)),
                            UBUNTU: ppv((UBUNTU, "20.04")),
                        }
                    ),
                )
            )
        for gcc_version in [1, 3, 6, 7, 13, 99]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD({HOST_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "18.04"))}),
                )
            )
        for gcc_version in [1, 3, 6, 7, 13, 99]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD({DEVICE_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "18.04"))}),
                )
            )
        for gcc_version in [1, 3, 6, 7, 13, 99]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            DEVICE_COMPILER: ppv((GCC, gcc_version)),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                )
            )
        for gcc_version in [1, 3, 6, 7, 13, 99]:
            self.assertTrue(
                software_dependency_filter_typechecked(
                    OD(
                        {
                            HOST_COMPILER: ppv((GCC, gcc_version)),
                            UBUNTU: ppv((UBUNTU, "18.04")),
                        }
                    ),
                )
            )

    def test_not_valid_gcc_is_in_ubuntu_2004_d1(self):
        for gcc_version in [6, 3, 1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD({HOST_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "20.04"))}),
                    reason_msg,
                ),
                f"host compiler GCC {gcc_version} + Ubuntu 20.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"host compiler GCC {gcc_version} is not available in Ubuntu 20.04",
            )
        for gcc_version in [6, 3, 1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD({HOST_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "22.04"))}),
                    reason_msg,
                ),
                f"host compiler GCC {gcc_version} + Ubuntu 22.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"host compiler GCC {gcc_version} is not available in Ubuntu 22.04",
            )

        for gcc_version in [6, 3, 1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD({DEVICE_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "20.04"))}),
                    reason_msg,
                ),
                f"device compiler GCC {gcc_version} + Ubuntu 20.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"device compiler GCC {gcc_version} is not available in Ubuntu 20.04",
            )
        for gcc_version in [6, 3, 1]:
            reason_msg = io.StringIO()
            self.assertFalse(
                software_dependency_filter_typechecked(
                    OD({DEVICE_COMPILER: ppv((GCC, gcc_version)), UBUNTU: ppv((UBUNTU, "22.04"))}),
                    reason_msg,
                ),
                f"device compiler GCC {gcc_version} + Ubuntu 22.04",
            )
            self.assertEqual(
                reason_msg.getvalue(),
                f"device compiler GCC {gcc_version} is not available in Ubuntu 22.04",
            )
