# pylint: disable=missing-docstring
import unittest
import packaging.version

from bashi.printer import ubuntu_version_to_string


class TestUbuntuVersionToString(unittest.TestCase):
    def test_ubuntu_version_to_string(self):
        self.assertEqual(ubuntu_version_to_string(packaging.version.parse("18.04")), "18.04")
        self.assertEqual(ubuntu_version_to_string(packaging.version.parse("22.04")), "22.04")
        self.assertEqual(ubuntu_version_to_string(packaging.version.parse("26.04")), "26.04")
        self.assertEqual(ubuntu_version_to_string(packaging.version.parse("26.10")), "26.10")
