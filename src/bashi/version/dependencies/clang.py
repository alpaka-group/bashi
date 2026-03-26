"""Contains relationships with Clang and other parameter-values."""

from typing import List
from bashi.version.dependencies.base_version_support import CompilerCxxSupport

# define the maximum supported cxx version for a specific clang version
CLANG_CXX_SUPPORT_VERSION: List[CompilerCxxSupport] = [
    CompilerCxxSupport("9", "17"),
    CompilerCxxSupport("14", "20"),
    CompilerCxxSupport("17", "23"),
]
