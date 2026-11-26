"""Dumper for pyyaml which enables multiline commands in the script section."""

import yaml


class MultilineYamlString(str):
    """Wrapper class to write multiline command."""


class MultilineDumper(yaml.SafeDumper):
    """Yaml Dumper for multiline commands."""


def represent_literal_string(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")


MultilineDumper.add_representer(MultilineYamlString, represent_literal_string)
