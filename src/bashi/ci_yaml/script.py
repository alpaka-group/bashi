"""Get script section for GitLab CI jobs."""

from collections.abc import Callable

from typeguard import typechecked
from bashi import ValueName
from bashi.printer import ubuntu_version_to_string
from bashi.spack import SpackAliasDict, get_spack_install_string
from bashi.types import ValueVersion


@typechecked
def get_spack_build_cache_url(ubuntu_version: ValueVersion) -> str:
    """
    Get path to default spack build cache for alpaka based projects.

    Args:
        ubuntu_version (ValueVersion): Each build cache is build for a specific ubuntu version.

    Returns:
        str: Return OCI container registry URL to the build cache
    """
    return (
        "oci://registry.hzdr.de/crp/alpaka-spack-buildcache/"
        f"cache-ubuntu{ubuntu_version_to_string(ubuntu_version).replace('.', '')}"
    )


@typechecked
def get_spack_build_script(
    ubuntu_version: ValueVersion,
    software_name: ValueName,
    software_version: ValueVersion,
    spack_alias_dict: SpackAliasDict,
    build_cache_url_func: Callable[[ValueVersion], str] | None = None,
) -> list[str]:
    """
    Get the script section of a spack package build and push to build cache CI job.

    Args:
        ubuntu_version (ValueVersion): Ubuntu version of the build cache
        software_name (ValueName): Value-name of the software to build. For example CLANG to build
            the llvm spack package.
        software_version (ValueVersion): Value-version of the spack package to be build.
        spack_alias_dict (SpackAliasDict): The spack alias dict maps the value-name and
            value-version to a spack package name and version.
        build_cache_url_func (Callable[[ValueVersion], str] | None): Functor to obtain the build
            cache URL depending on the given Ubuntu version. If it None, use
            bashi.get_spack_build_cache_url().

    Returns:
        list[str]: GitLab CI script section
    """
    build_push_instructions = get_spack_install_string(
        software_name, software_version, spack_alias_dict
    )
    script: list[str] = []

    if build_cache_url_func:
        build_cache_url = build_cache_url_func(ubuntu_version)
    else:
        build_cache_url = get_spack_build_cache_url(ubuntu_version)

    for build_instr in build_push_instructions.build:
        script.append(
            f"spack install {build_instr} "
            f"arch='linux-ubuntu{ubuntu_version_to_string(ubuntu_version)}-x86_64_v3'"
        )

    script.append(
        "spack mirror add "
        "--oci-username-variable REGISTRY_USER "
        "--oci-password-variable REGISTRY_TOKEN "
        "gitlabci "
        f"{build_cache_url}"
    )

    for push_instr in build_push_instructions.push:
        script.append(f"spack buildcache push gitlabci {push_instr}")

    script.append(f"spack buildcache update-index {build_cache_url}")

    return script
