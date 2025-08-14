from typing import Dict
import packaging.version
from bashi import Parameter, ValueName, ValueVersion

# CMake build type
BUILD_TYPE: Parameter = "build_type"
CMAKE_RELEASE: str = "0"
CMAKE_DEBUG: str = "1"
CMAKE_RELEASE_VER: ValueVersion = packaging.version.parse(CMAKE_RELEASE)
CMAKE_DEBUG_VER: ValueVersion = packaging.version.parse(CMAKE_DEBUG)
BUILD_TYPES_NAMES = {
    "Release": CMAKE_RELEASE_VER,
    "Debug": CMAKE_DEBUG_VER,
}


def get_version_aliases() -> Dict[ValueName, Dict[ValueVersion, str]]:
    """Returns a dict, which contains alias strings for specific value-versions of a specific
    parameter-value.

    Internal bashi works only with version numbers. Therefore if we want to use an string as
    value-version, we need to map the string to a version.

    Returns:
        Dict[ValueName, Dict[ValueVersion, str]]: string aliases for value-versions of
            parameter-values
    """
    version_aliases: Dict[ValueName, Dict[ValueVersion, str]] = {}
    for val_name, version_map in [(BUILD_TYPE, BUILD_TYPES_NAMES)]:
        version_map_parsed: Dict[ValueVersion, str] = {}
        for alias, ver in version_map.items():
            version_map_parsed[ver] = alias
        version_aliases[val_name] = version_map_parsed

    return version_aliases
