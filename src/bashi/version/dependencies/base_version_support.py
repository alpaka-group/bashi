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


# pylint: disable=too-few-public-methods
class CompilerCxxSupport(VersionSupportBase):
    """Contains a compiler version and host compiler version. Does automatically parse the input
    strings to package.version.Version.

    Provides comparision operators for sorting.
    """

    def __init__(self, compiler_version: str, cxx_version: str):
        VersionSupportBase.__init__(self, compiler_version, cxx_version)
        self.compiler: packaging.version.Version = self.version1
        self.cxx: packaging.version.Version = self.version2

    def __str__(self) -> str:
        return f"compiler {str(self.compiler)} + CXX {self.cxx}"


# pylint: disable=too-few-public-methods
class ClangBase(VersionSupportBase):
    """Contains a compiler version and Clang version which the compiler based on. Does automatically
    parse the input strings to package.version.Version.

    Provides comparision operators for sorting.
    """

    def __init__(self, compiler: str, clang: str):
        VersionSupportBase.__init__(self, compiler, clang)
        self.compiler: packaging.version.Version = self.version1
        self.clang: packaging.version.Version = self.version2

    def __str__(self) -> str:
        return f"Compiler {str(self.compiler)} + Clang {self.clang}"
