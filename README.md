# bashi
A library to provide a job generator for CI's for alpaka based projects.

# Developing

It is strongly recommended to use a Python environment for developing the code, such as `virtualenv` or a `conda` environment. The following code uses a `virtualenv`.

1. Create the environment: `virtualenv -p python3 env`
2. Activate the environment: `source env/bin/activate`
3. Install the library: `pip install --editable .`
4. Test the installation with the example: `python3 example/example.py`
5. You can run the unit tests by going to the `test` directory and running `python -m unittest`

If the example works correctly, a `job.yml` will be created in the current directory. You can also run `python3 example/example.py --help` to see additional options.

Now the library is available in the environment. Therefore you can easily use the library in your own projects via `import bashi`.

## Formatting the source code

The source code is formatted using the [black](https://pypi.org/project/black/) formatter and the default style guide. You must install it and run `black /path/to/file` to format a file. A CI job checks that all files are formatted correctly. If the job fails, a PR cannot be merged.

## Contribution

This section contains some hints for developing new functions. The hints are mainly for people who have no experience with `setuptools` and building `pip` packages.

* The `pip install --editable .` command installs the source code files as a symbolic link. This means that changes in the source code of the Python files in the `src/bashi` folder are directly available without any additional installation step (only a restart of the Python process/interpreter is required).
* The software requirements are defined in `pyproject.toml` and not in an additional `requirements.txt`.
* It is necessary to increase the version number in `version.txt` before a new feature can be merged in the master branch. Otherwise the upload to pypy.org will fail because existing versions cannot be changed.
