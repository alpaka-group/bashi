#!/usr/bin/env python3

"""Command line tool to check filter rules."""

import sys
import bashiValidate
from bashi import VersionRelation


def main() -> None:
    """Entry point for the application."""
    validator = bashiValidate.Validator(VersionRelation())
    sys.exit(int(not validator.validate()))


if __name__ == "__main__":
    main()
