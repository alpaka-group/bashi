"""Contains relationships with hipcc and other parameter-values."""

from typing import List
from bashi.version.dependencies.base_version_support import ClangBase

# This list stores which HIPCC version based on which Clang
# The list allows to reuse the knowledge of Clang and apply it on HIPCC like the C++ standard
# support.
HIPCC_CLANG_VERSION: List[ClangBase] = [
    ClangBase("5.1", "14"),
    ClangBase("5.2", "14"),
    ClangBase("5.3", "15"),
    ClangBase("5.5", "16"),
    ClangBase("5.6", "16"),
    ClangBase("5.7", "17"),
    ClangBase("6.0", "17"),
    ClangBase("6.1", "17"),
    ClangBase("6.2", "18"),
    ClangBase("6.3", "18"),
    ClangBase("6.4", "19"),
    ClangBase("7.0", "20"),
    ClangBase("7.1", "20"),
    ClangBase("7.2", "22"),
]
