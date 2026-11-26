"""Provide functionality to check if spack packages are available in a build cache for the given
required software versions."""

import subprocess
from collections.abc import Callable

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
    OFF_VER,
    UBUNTU,
)

from bashi.spack.types import OsPackageList, SpackAlias, SpackAliasDict, SpackPackagesList, pkg_iter
from bashi.utils import run_subprocess
from bashi.types import ValueVersion


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
