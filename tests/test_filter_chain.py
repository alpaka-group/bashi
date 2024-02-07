# pylint: disable=missing-docstring
import unittest
from typing import Dict, List
from collections import OrderedDict
import packaging.version as pkv
from bashi.types import ParameterValue, ParameterValueTuple, FilterFunction
from bashi.utils import get_default_filter_chain, FilterAdapter


class TestFilterChain(unittest.TestCase):
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

    def test_filter_chain_default(self):
        filter_chain: FilterFunction = get_default_filter_chain()
        self.assertTrue(
            filter_chain(self.param_val_tuple),
            "The filter should return true every time, "
            "because the test data should no trigger any rule",
        )

    def test_filter_chain_custom_filter_pass(self):
        def custom_filter(row: ParameterValueTuple):
            if "paramNotExist" in row:
                return False
            return True

        filter_chain: FilterFunction = get_default_filter_chain(custom_filter)
        self.assertTrue(
            filter_chain(self.param_val_tuple),
            "The production filters should return True all the time, because the test data set "
            "does not contain any production data. The custom filter should not match the test "
            "data.",
        )

    def test_filter_chain_custom_filter_match(self):
        def custom_filter(row: ParameterValueTuple):
            if "param2" in row:
                return False
            return True

        filter_chain: FilterFunction = get_default_filter_chain(custom_filter)
        self.assertFalse(
            filter_chain(self.param_val_tuple),
            "The production filters should return True all the time, because the test data set "
            "does not contain any production data. The custom filter should match the test data.",
        )

    def test_filter_chain_filter_adapter_default(self):
        filter_chain: FilterFunction = get_default_filter_chain()
        adapter = FilterAdapter(self.param_map, filter_chain)

        self.assertTrue(
            adapter(self.test_row),
            "The filter should return true every time, "
            "because the test data should no trigger any rule",
        )

    def test_filter_chain_filter_adapter_custom_filter_pass(self):
        def custom_filter(row: ParameterValueTuple):
            if "paramNotExist2" in row:
                return False
            return True

        filter_chain: FilterFunction = get_default_filter_chain(custom_filter)
        adapter = FilterAdapter(self.param_map, filter_chain)

        self.assertTrue(
            adapter(self.test_row),
            "The production filters should return True all the time, because the test data set "
            "does not contain any production data. The custom filter should not match the test "
            "data.",
        )

    def test_filter_chain_filter_adapter_custom_filter_match(self):
        def custom_filter(row: ParameterValueTuple):
            if "param1" in row:
                return False
            return True

        filter_chain: FilterFunction = get_default_filter_chain(custom_filter)
        adapter = FilterAdapter(self.param_map, filter_chain)

        self.assertFalse(
            adapter(self.test_row),
            "The production filters should return True all the time, because the test data set "
            "does not contain any production data. The custom filter should match the test data.",
        )
