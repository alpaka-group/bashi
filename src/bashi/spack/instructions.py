"""Spack instructions to build and push spack packages to the build cache."""

from collections.abc import Callable
import re
import subprocess
from bashi import ICPX, ValueName, ValueVersion
from bashi.globals import CLANG, GCC, HIPCC
from bashi.spack.types import SpackAliasDict, SpackBuildPushInstructions
from bashi.utils import run_subprocess


def get_spack_install_string(
    name: ValueName,
    version: ValueVersion,
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
