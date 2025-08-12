"""Example for a validator application with custom filter."""

import sys
import bashiValidate
from src.example_filter import ExampleFilter


if __name__ == "__main__":
    validator = bashiValidate.Validator()
    validator.add_software_version_parameter(name="SoftwareA", help_text="SoftwareA version number")
    validator.add_known_version(name="SoftwareA", versions=["1.0", "2.0", "2.1"])
    validator.add_custom_filter(ExampleFilter())
    sys.exit(int(not validator.validate()))
