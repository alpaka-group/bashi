"""Get script section for GitLab CI jobs."""

from collections.abc import Callable

from typeguard import typechecked
from bashi import ValueName
from bashi.printer import ubuntu_version_to_string
from bashi.spack.types import SpackAliasDict
from bashi.spack.instructions import get_spack_install_string
from bashi.types import ValueVersion
from bashi.ci_yaml.dumper import MultilineYamlString


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
    spack_package_repository_sha: str,
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
        spack_package_repository_sha (str): Git commit hash of the builtin spack package
            repository. If empty, does not update spack package repository and use
            preinstalled version.
        build_cache_url_func (Callable[[ValueVersion], str] | None): Functor to obtain the build
            cache URL depending on the given Ubuntu version. If it None, use
            bashi.get_spack_build_cache_url().

    Returns:
        list[str]: GitLab CI script section
    """
    build_push_instructions = get_spack_install_string(
        software_name, software_version, spack_alias_dict
    )
    script: list[str] = [". /opt/spack/share/spack/setup-env.sh"]

    if spack_package_repository_sha != "":
        script += [
            "_spack_packages_repo_path=$(spack repo list | awk '$2 == \"builtin\" {print $NF}')",
            '_spack_packages_repo_commit_hash=$(git -C "${_spack_packages_repo_path}" rev-parse HEAD)',
            MultilineYamlString(
                'if [[ "${_spack_packages_repo_commit_hash}" '
                f'!= "{spack_package_repository_sha}" ]]; then\n'
                '   echo -e "\\e[1;33mUpdate to spack package repository to git commit hash '
                '${_spack_packages_repo_commit_hash}\\e[0m"\n'
                '   spack repo update -c "${ACPI_SPACK_PACKAGES_REPO_HASH}" builtin\n'
                "fi"
            ),
        ]

    script.append(
        MultilineYamlString(
            "retry_cmd() {\n"
            "   (\n"
            "       set +euo pipefail\n"
            "       i=0\n"
            '       while [ "$i" -lt 3 ]; do\n'
            '           "$@"\n'
            '           result="$?"\n'
            '           if [[ "$result" -eq 0 ]]; then\n'
            "               return 0\n"
            "           fi\n"
            "           echo \"[WARNING]: Attempt #${i} to run '$*' failed\"\n"
            "           sleep 3\n"
            "           i=$(( i + 1 ))\n"
            "       done\n"
            '       return "$result"\n'
            "   )\n"
            "}"
        )
    )

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
        "--oci-username-variable AP_REGISTRY_USER "
        "--oci-password-variable AP_REGISTRY_TOKEN "
        "gitlabci "
        f"{build_cache_url}"
    )

    for push_instr in build_push_instructions.push:
        script.append(f"retry_cmd spack buildcache push gitlabci {push_instr}")

    script.append(f"retry_cmd spack buildcache update-index {build_cache_url}")

    return script
