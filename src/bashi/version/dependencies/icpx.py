"""Contains relationships with icpx and other parameter-values."""

from typing import List
from bashi.version.dependencies.base_version_support import ClangBase


# This list stores which ICPX version based on which Clang
# The list allows to reuse the knowledge of Clang and apply it on ICPX like the C++ standard
# support.
ICPX_CLANG_VERSION: List[ClangBase] = [ClangBase("2025.0", "19")]
