# bashi
A library to provide a job generator for CI's for alpaka based projects.

## Contribution

This section contains some hints for developing new functions. The hints are mainly for people who have no experience with `setuptools` and building `pip` packages.

* The `pip install --editable .` command installs the source code files as a symbolic link. This means that changes in the source code of the Python files in the `src/bashi` folder are directly available without any additional installation step (only a restart of the Python process/interpreter is required).
* The software requirements are defined in `pyproject.toml` and not in an additional `requirements.txt`.
* It is necessary to increase the version number in `version.txt` before a new feature can be merged in the master branch. Otherwise the upload to pypy.org will fail because existing versions cannot be changed.
