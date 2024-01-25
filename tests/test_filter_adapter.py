# pylint: disable=missing-docstring
import unittest
from typing import Tuple, Dict, List
from collections import OrderedDict
from packaging.version import Version
import packaging.version as pkv
from typeguard import typechecked
from bashi.types import ParameterValueTuple, ParameterValue
from bashi.utils import FilterAdapter


class TestFilterAdapterDataSet1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.param_val_tuple: ParameterValueTuple = OrderedDict()
        cls.param_val_tuple["param1"] = ("param-val-name1", pkv.parse("1"))
        cls.param_val_tuple["param2"] = ("param-val-name2", pkv.parse("2"))
        cls.param_val_tuple["param3"] = ("param-val-name3", pkv.parse("3"))

        cls.param_map: Dict[int, str] = {}
        for index, param_name in enumerate(cls.param_val_tuple.keys()):
            cls.param_map[index] = param_name

        cls.test_row: List[ParameterValue] = []
        for param_val in cls.param_val_tuple.values():
            cls.test_row.append(param_val)

    # use typechecked to do a deep type check
    # isinstance() only verify the "outer" data type, which is OrderedDict
    # isinstance() does not verify the key and value type
    def test_function_type(self):
        @typechecked
        def filter_function(row: ParameterValueTuple) -> bool:
            if len(row.keys()) < 1:
                raise AssertionError("There is no element in row.")

            # typechecked does not check the types of Tuple, therefore I "unwrap" it
            @typechecked
            def check_param_value_type(_: ParameterValue):
                pass

            check_param_value_type(next(iter(row.values())))

            return True

        filter_adapter = FilterAdapter(self.param_map, filter_function)
        self.assertTrue(filter_adapter(self.test_row))

    def test_function_length(self):
        def filter_function(row: ParameterValueTuple) -> bool:
            if len(row) != 3:
                raise AssertionError(f"Size of test_row is {len(row)}. Expected is 3.")

            return True

        filter_adapter = FilterAdapter(self.param_map, filter_function)
        self.assertTrue(filter_adapter(self.test_row))

    def test_function_row_order(self):
        def filter_function(row: ParameterValueTuple) -> bool:
            excepted_param_order = ["param1", "param2", "param3"]
            if len(excepted_param_order) != len(row):
                raise AssertionError(
                    "excepted_key_order and row has not the same length.\n"
                    f"{len(excepted_param_order)} != {len(row)}"
                )

            for index, param in enumerate(row.keys()):
                if excepted_param_order[index] != param:
                    raise AssertionError(
                        f"The {index}. parameter is not the expected "
                        f"parameter: {excepted_param_order[index]}"
                    )

            expected_param_value_order = [
                ("param-val-name1", pkv.parse("1")),
                ("param-val-name2", pkv.parse("2")),
                ("param-val-name3", pkv.parse("3")),
            ]

            for index, param_value in enumerate(row.values()):
                expected_value_name = expected_param_value_order[index][0]
                expected_value_version = expected_param_value_order[index][1]
                if (
                    expected_value_name != param_value[0]
                    or expected_value_version != param_value[1]
                ):
                    raise AssertionError(
                        f"The {index}. parameter-value is not the expected parameter-value\n"
                        f"Get: {param_value}\n"
                        f"Expected: {expected_param_value_order[index]}"
                    )

            return True

        filter_adapter = FilterAdapter(self.param_map, filter_function)
        self.assertTrue(filter_adapter(self.test_row))

    def test_lambda(self):
        filter_adapter = FilterAdapter(self.param_map, lambda row: len(row) == 3)
        self.assertTrue(filter_adapter(self.test_row), "row has not the length of 3")


# do a complex test with a different data set
class TestFilterAdapterDataSet2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.param_val_tuple: ParameterValueTuple = OrderedDict()
        cls.param_val_tuple["param6b"] = ("param-val-name1", pkv.parse("3.21.2"))
        cls.param_val_tuple["param231a"] = ("param-val-name67asd", pkv.parse("2.4"))
        cls.param_val_tuple["param234s"] = ("param-val-678", pkv.parse("3"))
        cls.param_val_tuple["foo"] = ("foo", pkv.parse("12.3"))
        cls.param_val_tuple["bar"] = ("bar", pkv.parse("3"))

        cls.param_map: Dict[int, str] = {}
        for index, param_name in enumerate(cls.param_val_tuple.keys()):
            cls.param_map[index] = param_name

        cls.test_row: List[Tuple[str, Version]] = []
        for param_val in cls.param_val_tuple.values():
            cls.test_row.append(param_val)

    def test_function_row_length_order(self):
        def filter_function(row: ParameterValueTuple) -> bool:
            excepted_param_order = ["param6b", "param231a", "param234s", "foo", "bar"]
            if len(excepted_param_order) != len(row):
                raise AssertionError(
                    "excepted_key_order and row has not the same length.\n"
                    f"{len(excepted_param_order)} != {len(row)}"
                )

            for index, param in enumerate(row.keys()):
                if excepted_param_order[index] != param:
                    raise AssertionError(
                        f"The {index}. parameter is not the expected "
                        f"parameter: {excepted_param_order[index]}"
                    )

            expected_param_value_order = [
                ("param-val-name1", pkv.parse("3.21.2")),
                ("param-val-name67asd", pkv.parse("2.4")),
                ("param-val-678", pkv.parse("3")),
                ("foo", pkv.parse("12.3")),
                ("bar", pkv.parse("3")),
            ]

            for index, param_value in enumerate(row.values()):
                expected_value_name = expected_param_value_order[index][0]
                expected_value_version = expected_param_value_order[index][1]
                if (
                    expected_value_name != param_value[0]
                    or expected_value_version != param_value[1]
                ):
                    raise AssertionError(
                        f"The {index}. parameter-value is not the expected parameter-value\n"
                        f"Get: {param_value}\n"
                        f"Expected: {expected_param_value_order[index]}"
                    )

            return True

        filter_adapter = FilterAdapter(self.param_map, filter_function)
        self.assertTrue(filter_adapter(self.test_row))
