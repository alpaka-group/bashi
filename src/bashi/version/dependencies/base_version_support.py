"""Base class to store value-version pairs."""

import packaging.version


class VersionSupportBase:
    """Contains two value-versions. Does automatically parse the input strings to
    package.version.Version.

    Provides comparision operators for sorting.
    """

    def __init__(self, version1: str, version2: str):
        self.version1 = packaging.version.parse(version1)
        self.version2 = packaging.version.parse(version2)

    def __lt__(self, other: "VersionSupportBase") -> bool:
        return self.version1 < other.version1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise TypeError(f"does not support other types than {type(self).__name__}")
        return self.version1 == other.version1 and self.version2 == other.version2
