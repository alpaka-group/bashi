"""Get image for GitLab CI jobs."""

from typeguard import typechecked
from bashi.printer import ubuntu_version_to_string
from bashi.types import ValueVersion


@typechecked
def get_spack_build_image_name(ubuntu_version: ValueVersion, image_tag: str) -> str:
    """
    Get the base image to build new spack packages for the build cache.

    Args:
        ubuntu_version (ValueVersion): Ubuntu version of the base image.
        image_tag (str): Image tag

    Returns:
        str: Image URL
    """
    return (
        "registry.hzdr.de/crp/"
        "alpaka-spack-buildcache/"
        f"alpaka-ci-spack-ubuntu{ubuntu_version_to_string(ubuntu_version)}"
        f":{image_tag}"
    )
