"""Provide functionality to check if spack packages are available in a build cache for the given
required software versions."""

import re
import subprocess
from collections.abc import Callable, Iterator
from itertools import chain

import bashi
from bashi.globals import (
    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
    ALPAKA_ACC_GPU_CUDA_ENABLE,
    CLANG,
    CLANG_CUDA,
    CMAKE,
    COMPILERS,
    DEVICE_COMPILER,
    GCC,
    HIPCC,
    HOST_COMPILER,
    ICPX,
    OFF_VER,
    UBUNTU,
)

from bashi.utils import run_subprocess
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
SpackAliasDict = dict[bashi.ValueName, SpackAlias]

# For each Ubuntu version, store a list of software package versions
OsPackageList = dict[bashi.ValueVersion, dict[bashi.ValueName, list[bashi.ValueVersion]]]


def pkg_iter(
    data: dict[bashi.ValueName, list[bashi.ValueVersion]],
) -> Iterator[tuple[bashi.ValueName, bashi.ValueVersion]]:
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
) -> Iterator[tuple[bashi.ValueVersion, bashi.ValueName, bashi.ValueVersion]]:
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


def get_spack_alias() -> SpackAliasDict:
    """
    Return a dict, which maps bashi value-names to spack package names.

    Returns:
        Dict[ValueName, str]: Dictionary with value-name spack package name mapping.
    """
    return {
        GCC: SpackAlias("gcc"),
        CLANG: SpackAlias("llvm"),
        HIPCC: SpackAlias("hip"),
        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE: SpackAlias("intel-oneapi-tbb"),
        ALPAKA_ACC_GPU_CUDA_ENABLE: SpackAlias("cuda"),
        CMAKE: SpackAlias("cmake"),
    }


class SpackBuilder:
    """Calculate, which spack packages needs to be built for a given combination list and be pushed
    to the build cache.

    The various getter functions rely on the results of other getter functions. If a getter B
    depends on the result of a getter A, it automatically calls getter A to retrieve the results.
    All getter results are automatically cached. This assumes that the local Spack installation and
    the remote build cache remain unchanged.

    Some functions calls the external tool spack.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        combination_list: bashi.CombinationList,
        spack_packages: SpackAliasDict,
        build_cache_url_functor: Callable[[ValueVersion], str],
    ):
        """
        Set up the input for the calculation. A value name will be used to calculate the required
        spack packages only if it appears in both the combination_list and spack_packages.

        Attentions: The value-name ICPX is not supported because of License reasons of the OneAPI
        SDK.

        Args:
            combination_list (bashi.CombinationList): combination-list
            spack_packages (SpackAliasDict): The spack alias dict, which all spack package names,
                which are related to a specific value-name.
            build_cache_url_functor (Callable[[ValueVersion], ValueVersion] | None): The functor
                takes a ubuntu version as input an return the URL of the build cache.
        """
        self.combination_list = combination_list
        self.spack_packages = spack_packages
        self.build_cache_url_functor = build_cache_url_functor

        self.cached_required_packages: OsPackageList = {}
        self.cached_available_spack_packages: dict[bashi.ValueName, list[str]] = {}
        self.cached_buildable_spack_packages: SpackPackagesList = SpackPackagesList()
        self.cached_buildcache_packages: SpackPackagesList = SpackPackagesList()

        # Use reference to run_process function instead calling function to allow to mock up the
        # process in the unit tests.
        self._run_process: Callable[[list[str]], subprocess.CompletedProcess[str]] = run_subprocess

    def reset(self) -> None:
        """A SpackBuilder object caches different intermediate results of the calculation process of
        the required spack packages to avoid expensive external calls of spack. Reset the caches.
        """
        self.cached_required_packages = {}
        self.cached_available_spack_packages = {}
        self.cached_buildable_spack_packages = SpackPackagesList()
        self.cached_buildcache_packages = SpackPackagesList()

    @staticmethod
    def _package_is_in_list(name: str, package_list: list[str]) -> bool:
        """
        Check if a package is in the list. Package in the package_list does not need to match name
        exactly.

        Args:
            name (str): Name of the search package.
            package_list (list[str]): Package list

        Returns:
            bool: Return true, if exist
        """
        return len([pkg for pkg in package_list if pkg.startswith(name)]) > 0

    @staticmethod
    def _dict_add_unique_list_entry(key, value, d: dict) -> None:
        """
        For a given key `key`, add `value` to list, where `key` is pointing to it. If `key` does
        not exist in `d`, create an empty list before add `value`. If `value` is already in the
        list, don't add it.

        Args:
            key (type): Key name of the entry
            value (type): Value which needs to be added to the list, where key is pointing to it.
            d (dict): Dict where key and value are added.
        """
        if key not in d:
            d[key] = []
        if value not in d[key]:
            d[key].append(value)

    def get_required_packages(self) -> OsPackageList:
        """
        Search for required spack packages in the combinations list. Does not check if a spack
        package or a specific version of spack package exist.

        Returns:
            OsPackageList: Return a list of required spack packages for each Ubuntu version.
        """
        if self.cached_required_packages:
            return self.cached_required_packages

        def add_required_spack_compiler_packages(
            combination: bashi.Combination,
            required_packages: dict[bashi.ValueName, list[bashi.ValueVersion]],
        ):
            used_compilers = set(self.spack_packages.keys()).intersection(set(COMPILERS))

            # Process the special case nvcc + gcc/clang. Therefore we need to check the host and
            # device compiler. In all other cases, the host and device compiler are the same.
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                if combination[compiler_type].name in used_compilers:
                    # There is no special "clang-cuda" spack packages. The llvm spack packages is
                    # used for clang and clang-cuda.
                    compiler_alias = (
                        CLANG
                        if combination[compiler_type].name == CLANG_CUDA
                        else combination[compiler_type].name
                    )
                    sw_version = self.spack_packages[compiler_alias].version_alias(
                        comb[compiler_type].version
                    )
                    self._dict_add_unique_list_entry(compiler_alias, sw_version, required_packages)

        def add_required_spack_software_packages(
            combination: bashi.Combination,
            required_packages: dict[bashi.ValueName, list[bashi.ValueVersion]],
        ):
            used_software = set(self.spack_packages.keys()) - set(COMPILERS)

            for sw in used_software:
                if sw in combination:
                    if combination[sw].version == OFF_VER:
                        continue

                    self._dict_add_unique_list_entry(
                        sw,
                        self.spack_packages[sw].version_alias(combination[sw].version),
                        required_packages,
                    )

        for comb in self.combination_list:
            if comb[UBUNTU].version not in self.cached_required_packages:
                self.cached_required_packages[comb[UBUNTU].version] = {}

            add_required_spack_compiler_packages(
                comb, self.cached_required_packages[comb[UBUNTU].version]
            )
            add_required_spack_software_packages(
                comb, self.cached_required_packages[comb[UBUNTU].version]
            )

        return self.cached_required_packages

    # return a dict of all available spack package versions
    # the list contains versions which can be build
    # it does not mean, that the packages are available in the build cache
    def get_available_spack_packages(self) -> dict[bashi.ValueName, list[str]]:
        """
        Get all available versions for each spack package in SpackAliasDict. This spack packages
        can be built.

        Returns:
            dict[bashi.ValueName, list[str]]: Contains all available spack package version to the
                related value-name.
        """
        if self.cached_available_spack_packages:
            return self.cached_available_spack_packages

        for param_val, spack_name in {
            param_name: spack_alias.package_name
            for param_name, spack_alias in self.spack_packages.items()
        }.items():
            package_list_str = self._run_process(["spack", "versions", "--safe", spack_name]).stdout
            self.cached_available_spack_packages[param_val] = [
                pkg.strip() for pkg in package_list_str.split("\n") if pkg.strip() != ""
            ]

        return self.cached_available_spack_packages

    def get_required_available_spack_packages(self) -> SpackPackagesList:
        """
        Take the required packages from the combination-list and check if they can be build with
        spack.

        Returns:
            SpackPackagesList: The package list contains for each Ubuntu version the packages which
            can be built and a list of package which cannot be built with spack.
        """
        if self.cached_buildable_spack_packages:
            return self.cached_buildable_spack_packages

        existing_packages = self.cached_buildable_spack_packages.existing_packages
        missing_packages = self.cached_buildable_spack_packages.missing_packages

        for (
            ubuntu_ver,
            required_packages_for_ubuntu_version,
        ) in self.get_required_packages().items():
            if ubuntu_ver not in existing_packages:
                existing_packages[ubuntu_ver] = {}

            if ubuntu_ver not in missing_packages:
                missing_packages[ubuntu_ver] = {}

            for pkg_name, pkg_version in pkg_iter(required_packages_for_ubuntu_version):
                if self._package_is_in_list(
                    str(pkg_version), self.get_available_spack_packages()[pkg_name]
                ):
                    self._dict_add_unique_list_entry(
                        pkg_name, pkg_version, existing_packages[ubuntu_ver]
                    )
                else:
                    self._dict_add_unique_list_entry(
                        pkg_name, pkg_version, missing_packages[ubuntu_ver]
                    )

        return self.cached_buildable_spack_packages

    def get_build_cache_packages(
        self,
    ) -> SpackPackagesList:
        """
        Get all packages for each Ubuntu version, which are available in the build cache.

        Returns:
            SpackPackagesList: The package list contains for each Ubuntu version the packages which
            is already in the build cache and what is missing.
        """
        if self.cached_buildcache_packages:
            return self.cached_buildcache_packages

        existing_packages: OsPackageList = self.cached_buildcache_packages.existing_packages
        missing_packages: OsPackageList = self.cached_buildcache_packages.missing_packages
        for (
            ubuntu_ver,
            required_packages_for_ubuntu_version,
        ) in self.get_required_available_spack_packages().existing_packages.items():
            existing_packages[ubuntu_ver] = {}
            missing_packages[ubuntu_ver] = {}

            mirror_name = f"gitlabci-ubuntu{bashi.ubuntu_version_to_string(ubuntu_ver)}"
            mirror_url = self.build_cache_url_functor(ubuntu_ver)

            self._run_process(["spack", "mirror", "add", mirror_name, mirror_url])

            mirror_package_list_str = self._run_process(
                ["spack", "buildcache", "list", "--allarch"]
            ).stdout
            mirror_package_list = [
                pkg
                for pkg in mirror_package_list_str.split("\n")
                if pkg != "" and not pkg.startswith("--")
            ]

            for pkg_name, pkg_version in pkg_iter(required_packages_for_ubuntu_version):
                spack_package_name = f"{self.spack_packages[pkg_name].package_name}@{pkg_version}"
                if self._package_is_in_list(spack_package_name, mirror_package_list):
                    self._dict_add_unique_list_entry(
                        pkg_name, pkg_version, existing_packages[ubuntu_ver]
                    )
                else:
                    self._dict_add_unique_list_entry(
                        pkg_name, pkg_version, missing_packages[ubuntu_ver]
                    )

            self._run_process(["spack", "mirror", "rm", mirror_name])

        return self.cached_buildcache_packages


def get_spack_install_string(
    name: bashi.ValueName,
    version: bashi.ValueVersion,
    spack_alias: SpackAliasDict,
    run_process: Callable[[list[str]], subprocess.CompletedProcess[str]] = run_subprocess,
) -> SpackBuildPushInstructions:
    """
    Returns a list of spack instructions for a specific value-name and value-version, which can be
    used to build a package and push it to the build cache. Some value-names require more than one
    spack package. The build instructions also include build flags to optimize the build process,
    such as reducing the package size.

    The build instructions can placed in the spack install command:
        spack install <build_instruction>
    The push instructions can placed in the spack buildcache push command:
        spack buildcache push mirror_name <push_instruction>

    Args:
        name (bashi.ValueName): value-name
        version (bashi.ValueVersion): value-name
        spack_alias (SpackAliasDict): Alias dict to map value-name and value-version to a spack
            package.
        run_process (Callable[[list[str]], subprocess.CompletedProcess[str]]): Function hook to run
        external process. Is used to call spack executable (default: run_subprocess).

    Returns:
        SpackBuildPushInstructions: Object which contains build and push instructions.

    Raises:
        RuntimeError: value-name ICPX is not supported because of License issues of OneApi.
    """
    spack_package_name = f"{spack_alias[name].package_name}@{version}"
    if name == GCC:
        return SpackBuildPushInstructions(
            build_instructions=[f"{spack_package_name} +strip"],
            push_instructions=[spack_package_name],
        )

    if name == CLANG:
        return SpackBuildPushInstructions(
            build_instructions=[f"{spack_package_name} +link_llvm_dylib targets=x86,nvptx,amdgpu"],
            push_instructions=[spack_package_name],
        )

    if name == HIPCC:
        hip_dependencies = run_process(["spack", "spec", f"hip@{version}"]).stdout

        match = re.search(r"\^llvm@(\d+)\.", hip_dependencies)
        if match:
            llvm_version = match.group(1)
        else:
            raise RuntimeError(f"Cannot found llvm dependency of hip@{version}")

        build_instructions: list[str] = []
        push_instructions: list[str] = []

        build_instructions.append(f"llvm@{llvm_version} +link_llvm_dylib targets=x86,nvptx,amdgpu")
        build_instructions.append(f"hip@{version}")
        push_instructions.append(f"hip@{version}")

        build_instructions.append(f"hiprand@{version}")
        push_instructions.append(f"hiprand@{version}")

        build_instructions.append(f"hipcub@{version}")
        push_instructions.append(f"hipcub@{version}")

        # By default, hiprand is build for all GPU architecture.
        # A build of hiprand for all architecture has a size of about 500 MB.
        # A build for a single architecture has a size of about 80 MB.
        # Therefore build specific versions for the architectures, which we use in the CI.
        # The CI job will check, if a specific build is available and fall bak to the big package
        # in the case, the specific build does not exist.
        for gpu_arch in ["gfx90a", "gfx1030", "gfx1100"]:
            build_instructions.append(f"hiprand@{version} amdgpu_target={gpu_arch}")
            push_instructions.append(f"hiprand@{version} amdgpu_target={gpu_arch}")
            build_instructions.append(f"hipcub@{version} amdgpu_target={gpu_arch}")
            push_instructions.append(f"hipcub@{version} amdgpu_target={gpu_arch}")

        return SpackBuildPushInstructions(
            build_instructions=build_instructions, push_instructions=push_instructions
        )

    if name == ICPX:
        raise RuntimeError(
            "Cannot push OneAPI package (ICPX) to build cache.\n"
            "See: https://github.com/spack/spack/issues/53257"
        )

    return SpackBuildPushInstructions(
        build_instructions=[spack_package_name], push_instructions=[spack_package_name]
    )
