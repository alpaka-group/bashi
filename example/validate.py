"""Example for a validator application with custom filter."""

import sys
from typing import List
import bashiValidate
from src.example_filter import ExampleFilter
from src.globals import BUILD_TYPE, CMAKE_RELEASE, CMAKE_DEBUG, BUILD_TYPES_NAMES


def main(args: List[str], silent: bool) -> bool:
    """Main function

    Args:
        args (List[str]): Application argument list.
        silent (bool): disable terminal output

    Returns:
        bool: True if parameter-value-pair passed all filters
    """
    # setting args and silent are only required for the unit tests
    validator = bashiValidate.Validator(args=args, silent=silent)
    validator.add_software_version_parameter(
        name="SoftwareA", help_text="SoftwareA version number", short_name="SoftA"
    )
    validator.add_string_parameter(BUILD_TYPE, "CMake Build Type", BUILD_TYPES_NAMES, "buildType")
    validator.add_known_version(name="SoftwareA", versions=["1.0", "2.0", "2.1"])
    validator.add_known_version(name=BUILD_TYPE, versions=[CMAKE_RELEASE, CMAKE_DEBUG])
    validator.add_custom_filter(ExampleFilter())
    return validator.validate()


if __name__ == "__main__":
    sys.exit(int(not main(sys.argv[1:], False)))
