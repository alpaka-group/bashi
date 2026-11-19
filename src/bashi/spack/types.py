"""Spack related types."""

from collections.abc import Callable, Iterator
from itertools import chain

from bashi import ValueName
from bashi.types import ValueVersion


# pylint: disable=too-few-public-methods
class SpackAlias:
    """Stores the spack package name which is related to a bashi value-name. For example if the
    bashi.HOST_COMPILER bashi.CLANG is required, the required spack package is llvm.

    Is used mainly with a SpackAliasDict.
    """

    def __init__(
        self,
        spack_package_name: str,
        version_alias_functor: Callable[[ValueVersion], ValueVersion] | None = None,
    ):
        """
        Setup spack package names and an optional version alias.

        Args:
            spack_package_name (str): Spack package name.
            version_alias_functor (Callable[[ValueVersion], ValueVersion] | None): The functor
                takes a version number as input and returns a different version based on that. For
                example, if the Intel TBB backend is enabled (ON_VER → Version 1.0.0), version
                2023.10 is returned to install the intel-oneapi-tbb@2023.10 package. If the value
                is `None,` the input version is retained.
        """
        self.package_name = spack_package_name
        self.version_alias_functor = version_alias_functor

    def version_alias(self, version: ValueVersion) -> ValueVersion:
        """
        Depending on the version, return the same version or another version.

        Args:
            version (ValueVersion): Input version

        Returns:
            ValueVersion: Correct version
        """
        if self.version_alias_functor:
            return self.version_alias_functor(version)
        return version


# Maps a value-version to a related spack package name
SpackAliasDict = dict[ValueName, SpackAlias]

# For each Ubuntu version, store a list of software package versions
OsPackageList = dict[ValueVersion, dict[ValueName, list[ValueVersion]]]


# pylint: disable=too-few-public-methods
class SpackBuildPushInstructions:
    """Stores all spack package build instructions for a value-name. If a combination includes, for
    example, a HIP DEVICE_COMPILER, the required SPACK packages are hip, hiprand, and hipcub. It
    also stores additional spack build flags for the corresponding package. It further includes the
    appropriate commands to transfer the packages to the build cache in the correct order."""

    def __init__(
        self, build_instructions: list[str] | None, push_instructions: list[str] | None
    ) -> None:
        """
        Set build and push instructions.

        Args:
            build_instructions (list[str] | None): Build instructions. If set None, a empty list is
                setup.
            push_instructions (list[str] | None): Push instructions. If set None, a empty list is
                setup.
        """
        self.build: list[str] = []
        if build_instructions:
            self.build = build_instructions
        self.push: list[str] = []
        if push_instructions:
            self.push = push_instructions


# pylint: disable=too-few-public-methods
class SpackPackagesList:
    """Support object to store, if a spack package is existing and missing for a specific Ubuntu
    version. Can be used in different contexts."""

    def __init__(self) -> None:
        """Initialize with empty dicts."""
        self.existing_packages: OsPackageList = {}
        self.missing_packages: OsPackageList = {}

    def __bool__(self) -> bool:
        """
        Check if object is empty.

        Returns:
            bool: If all dicts are empty, return false.
        """
        return bool(self.existing_packages) and bool(self.missing_packages)


def pkg_iter(
    data: dict[ValueName, list[ValueVersion]],
) -> Iterator[tuple[ValueName, ValueVersion]]:
    """
    Creates an iterator, which iterates over all value-version of each value-name.

    Args:
        data (dict[bashi.ValueName, list[bashi.ValueVersion]]): Data

    Returns:
        Iterator[tuple[bashi.ValueName, bashi.ValueVersion]]: Iterator
    """
    return chain.from_iterable(
        ((name, version) for version in versions) for name, versions in data.items()
    )


def os_pkg_iter(
    data: OsPackageList,
) -> Iterator[tuple[ValueVersion, ValueName, ValueVersion]]:
    """
    Creates an iterator, which iterates over ubuntu versions first and the over all value-version
    of each value-name.

    Args:
        data (dict[bashi.ValueName, list[bashi.ValueVersion]]): Data

    Returns:
        Iterator[tuple[bashi.ValueName, bashi.ValueVersion]]: Iterator
    """
    return chain.from_iterable(
        ((ubuntu, pkg_name, pkg_version) for pkg_version in pkg_versions)
        for ubuntu, pkgs in data.items()
        for pkg_name, pkg_versions in pkgs.items()
    )
