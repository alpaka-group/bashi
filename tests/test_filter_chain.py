# pylint: disable=missing-docstring
import unittest
from typing import IO, Dict, List, Callable
from collections import OrderedDict
import packaging.version as pkv
from bashi.types import ParameterValue, ParameterValueTuple, FilterFunction
from bashi.filter import FilterBase
from bashi.filter_chain import get_default_filter_chain


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
        filter_chain: FilterBase = get_default_filter_chain()
        self.assertTrue(
            filter_chain(self.param_val_tuple),
            "The filter should return true every time, "
            "because the test data should no trigger any rule",
        )

    def test_filter_chain_custom_filter_pass(self):
        class CustomFilter(FilterBase):
            def __init__(
                self,
                runtime_infos: Dict[str, Callable[..., bool]] = {},
                output: IO[str] | None = None,
            ):
                super().__init__(runtime_infos, output)

            def __call__(self, row: ParameterValueTuple):
                if "paramNotExist" in row:
                    return False
                return True

        custom_filter = CustomFilter()

        filter_chain: FilterFunction = get_default_filter_chain(custom_filter=custom_filter)
        self.assertTrue(
            filter_chain(self.param_val_tuple),
            "The production filters should return True all the time, because the test data set "
            "does not contain any production data. The custom filter should not match the test "
            "data.",
        )

    def test_filter_chain_custom_filter_match(self):
        class CustomFilter(FilterBase):
            def __init__(
                self,
                runtime_infos: Dict[str, Callable[..., bool]] = {},
                output: IO[str] | None = None,
            ):
                super().__init__(runtime_infos, output)

            def __call__(self, row: ParameterValueTuple):
                if "param2" in row:
                    return False
                return True

        custom_filter = CustomFilter()

        filter_chain: FilterFunction = get_default_filter_chain(custom_filter=custom_filter)
        self.assertFalse(
            filter_chain(self.param_val_tuple),
            "The production filters should return True all the time, because the test data set "
            "does not contain any production data. The custom filter should match the test data.",
        )
