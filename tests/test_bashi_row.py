# pylint: disable=missing-docstring
import unittest
import operator
import packaging

from utils_test import parse_param_val as ppv
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.row import NonExistingEntry, NonExistingParameterValue, BashiRow


class TestBashiRow(unittest.TestCase):
    def test_comparision_non_existing_entry(self):
        non_existing_entry = NonExistingEntry()
        for logical_op in (
            operator.lt,
            operator.le,
            operator.eq,
            operator.ne,
            operator.gt,
            operator.ge,
        ):
            self.assertFalse(logical_op(non_existing_entry, "HIPCC"))
            self.assertFalse(logical_op(non_existing_entry, packaging.version.parse("3.0")))

    def test_comparision_non_existing_parameter_value(self):
        non_existing_parameter_value = NonExistingParameterValue()
        for logical_op in (
            operator.lt,
            operator.le,
            operator.eq,
            operator.ne,
            operator.gt,
            operator.ge,
        ):
            self.assertFalse(logical_op(non_existing_parameter_value.name, "HIPCC"))
            self.assertFalse(
                logical_op(non_existing_parameter_value.version, packaging.version.parse("3.0"))
            )

    def test_bashi_row_object(self):
        row = BashiRow({HOST_COMPILER: ppv((GCC, 13)), CMAKE: ppv((CMAKE, "3.25"))})

        self.assertTrue(HOST_COMPILER in row)
        self.assertTrue(CMAKE in row)
        self.assertFalse(DEVICE_COMPILER in row)

        self.assertTrue(row[HOST_COMPILER].name == GCC)
        self.assertTrue(row[HOST_COMPILER].version == packaging.version.parse(str(13)))
        self.assertTrue(row[CMAKE].name == CMAKE)
        self.assertTrue(row[CMAKE].version == packaging.version.parse("3.25"))

        self.assertFalse(row[HOST_COMPILER].name == CLANG)
        self.assertFalse(row[CMAKE].version == packaging.version.parse("3.15"))

        self.assertFalse(row[DEVICE_COMPILER].name == GCC)
        self.assertFalse(row[DEVICE_COMPILER].name == CLANG)
        self.assertFalse(row[DEVICE_COMPILER].version == packaging.version.parse(str(13)))
        self.assertFalse(row[DEVICE_COMPILER].version == packaging.version.parse("7.25"))

        # check if test statement without key-existing-check returns the same result as test
        # statement with key-existing-check
        self.assertEqual(
            row[HOST_COMPILER].name == GCC, HOST_COMPILER in row and row[HOST_COMPILER].name == GCC
        )
        self.assertEqual(
            row[HOST_COMPILER].name == CLANG,
            HOST_COMPILER in row and row[HOST_COMPILER].name == CLANG,
        )
        self.assertEqual(
            row[CMAKE].version == packaging.version.parse("3.25"),
            CMAKE in row and row[CMAKE].version == packaging.version.parse("3.25"),
        )
        self.assertEqual(
            row[CMAKE].version == packaging.version.parse("3.17"),
            CMAKE in row and row[CMAKE].version == packaging.version.parse("3.17"),
        )

        self.assertEqual(
            row[DEVICE_COMPILER].name == GCC,
            DEVICE_COMPILER in row and row[DEVICE_COMPILER].name == GCC,
        )
        self.assertEqual(
            row[DEVICE_COMPILER].name == CLANG,
            DEVICE_COMPILER in row and row[DEVICE_COMPILER].name == CLANG,
        )
        self.assertEqual(
            row[DEVICE_COMPILER].version == packaging.version.parse("3.25"),
            DEVICE_COMPILER in row
            and row[DEVICE_COMPILER].version == packaging.version.parse("3.25"),
        )
        self.assertEqual(
            row[DEVICE_COMPILER].version == packaging.version.parse("3.17"),
            DEVICE_COMPILER in row
            and row[DEVICE_COMPILER].version == packaging.version.parse("3.17"),
        )
