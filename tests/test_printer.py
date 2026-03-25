# pylint: disable=missing-docstring
import unittest
import packaging.version

from bashi.printer import ubuntu_version_to_string, on_off_ver_to_str
from bashi.globals import ON_VER, OFF_VER


class TestUbuntuVersionToString(unittest.TestCase):
    def test_ubuntu_version_to_string(self):
        self.assertEqual(ubuntu_version_to_string(packaging.version.parse("18.04")), "18.04")
        self.assertEqual(ubuntu_version_to_string(packaging.version.parse("22.04")), "22.04")
        self.assertEqual(ubuntu_version_to_string(packaging.version.parse("26.04")), "26.04")
        self.assertEqual(ubuntu_version_to_string(packaging.version.parse("26.10")), "26.10")


class TestOnOffVersionToString(unittest.TestCase):
    def test_on_off_version_to_string(self):
        self.assertEqual(on_off_ver_to_str(ON_VER), "ON")
        self.assertEqual(on_off_ver_to_str(OFF_VER), "OFF")

        with self.assertRaises(RuntimeError) as cm:
            on_off_ver_to_str(packaging.version.parse("42.0"))
        e = cm.exception
        self.assertEqual(str(e), "given version 42.0 is not ON_VER (1.0.0) or OFF_VER (0.0.0)")
