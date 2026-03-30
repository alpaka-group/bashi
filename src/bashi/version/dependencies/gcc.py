"""Contains relationships with GCC and other parameter-values."""

from typing import List
from bashi.version.dependencies.base_version_support import CompilerCxxSupport


# define the maximum supported cxx version for a specific gcc version
GCC_CXX_SUPPORT_VERSION: List[CompilerCxxSupport] = [
    CompilerCxxSupport("8", "17"),
    CompilerCxxSupport("10", "20"),
    CompilerCxxSupport("11", "23"),
]
