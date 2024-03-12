# pylint: disable=missing-docstring
import unittest
from typing import Dict, List
from collections import OrderedDict
import packaging.version as pkv
from typeguard import typechecked
from bashi.types import ParameterValueTuple, ParameterValue
from bashi.utils import FilterAdapter
from bashi.filter_compiler import compiler_filter_typechecked
from bashi.filter_backend import backend_filter
from bashi.filter_software_dependency import software_dependency_filter


class TestFilterAdapterDataSet1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.param_val_tuple: ParameterValueTuple = OrderedDict()
        cls.param_val_tuple["param1"] = ParameterValue("param-val-name1", pkv.parse("1"))
        cls.param_val_tuple["param2"] = ParameterValue("param-val-name2", pkv.parse("2"))
        cls.param_val_tuple["param3"] = ParameterValue("param-val-name3", pkv.parse("3"))

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
                ParameterValue("param-val-name1", pkv.parse("1")),
                ParameterValue("param-val-name2", pkv.parse("2")),
                ParameterValue("param-val-name3", pkv.parse("3")),
            ]

            for index, param_value in enumerate(row.values()):
                if (
                    expected_param_value_order[index].name != param_value.name
                    or expected_param_value_order[index].version != param_value.version
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

    # interface test with production filter
    def test_compiler_name_filter(self):
        error_msg = (
            "the filter should return true every time, "
            "because the test data should no trigger any rule"
        )
        self.assertTrue(
            FilterAdapter(self.param_map, compiler_filter_typechecked)(self.test_row),
            error_msg,
        )
        self.assertTrue(
            FilterAdapter(self.param_map, compiler_filter_typechecked)(self.test_row), error_msg
        )
        self.assertTrue(FilterAdapter(self.param_map, backend_filter)(self.test_row), error_msg)
        self.assertTrue(
            FilterAdapter(self.param_map, software_dependency_filter)(self.test_row), error_msg
        )


# do a complex test with a different data set
class TestFilterAdapterDataSet2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.param_val_tuple: ParameterValueTuple = OrderedDict()
        cls.param_val_tuple["param6b"] = ParameterValue("param-val-name1", pkv.parse("3.21.2"))
        cls.param_val_tuple["param231a"] = ParameterValue("param-val-name67asd", pkv.parse("2.4"))
        cls.param_val_tuple["param234s"] = ParameterValue("param-val-678", pkv.parse("3"))
        cls.param_val_tuple["foo"] = ParameterValue("foo", pkv.parse("12.3"))
        cls.param_val_tuple["bar"] = ParameterValue("bar", pkv.parse("3"))

        cls.param_map: Dict[int, str] = {}
        for index, param_name in enumerate(cls.param_val_tuple.keys()):
            cls.param_map[index] = param_name

        cls.test_row: List[ParameterValue] = []
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
                ParameterValue("param-val-name1", pkv.parse("3.21.2")),
                ParameterValue("param-val-name67asd", pkv.parse("2.4")),
                ParameterValue("param-val-678", pkv.parse("3")),
                ParameterValue("foo", pkv.parse("12.3")),
                ParameterValue("bar", pkv.parse("3")),
            ]

            for index, param_value in enumerate(row.values()):
                if (
                    expected_param_value_order[index].name != param_value.name
                    or expected_param_value_order[index].version != param_value.version
                ):
                    raise AssertionError(
                        f"The {index}. parameter-value is not the expected parameter-value\n"
                        f"Get: {param_value}\n"
                        f"Expected: {expected_param_value_order[index]}"
                    )

            return True

        filter_adapter = FilterAdapter(self.param_map, filter_function)
        self.assertTrue(filter_adapter(self.test_row))
