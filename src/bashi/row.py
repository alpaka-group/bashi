"""Row object for a bashi filter rule. See BashiRow class documentation."""

import covertable  # type: ignore
from bashi.types import ParameterValue


class NonExistingEntry:
    """A comparision with this object will return all the time false. That's the same result as a
    key does not exist in a dict."""

    def __lt__(self, _):
        return False

    def __le__(self, _):
        return False

    def __eq__(self, _):
        return False

    def __ne__(self, _):
        return False

    def __gt__(self, _):
        return False

    def __ge__(self, _):
        return False


# pylint: disable=too-few-public-methods
class NonExistingParameterValue:
    """Dummy interface object for a ParameterValue. Is used, if a parameter does not exist in a
    row."""

    def __init__(self) -> None:
        self.name = NonExistingEntry()
        self.version = NonExistingEntry()


class BashiRow(covertable.main.Row):
    """BashiRow provides the same interface as a `dict` or `OrderDict`, with one exception:
    if the access operator attempts to access a non-existent key, a dummy object is returned instead
    a error is thrown. The dummy object implements the `ParameterValue` interface and returns
    `false` whenever the member variables `name` and `version` are compared with another object.

    This eliminates the need to check whether a key exists in a filter rule:

    `if HOST_COMPILER in row and row[HOST_COMPILER].name == NVCC:`

    can be reduced to:

    `if row[HOST_COMPILER].name == NVCC:`
    """

    def __init__(self, row):
        if isinstance(row, covertable.main.Row):
            super().__init__(row, row.factors, row.serials, row.pre_filter)
        else:
            super().__init__(row, None, None, None)

    def __getitem__(self, item) -> ParameterValue | NonExistingParameterValue:
        if super().__contains__(item):
            return super().__getitem__(item)
        return NonExistingParameterValue()
