# pylint: disable=missing-docstring
import unittest
import copy
from bashi.versions import VERSIONS, get_parameter_value_matrix
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


class TestParameterValueGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.param_val_matrix = get_parameter_value_matrix()

    def test_all_params_in(self):
        expected_params = [HOST_COMPILER, DEVICE_COMPILER]
        expected_params += BACKENDS
        for sw_version in VERSIONS:
            if not sw_version in COMPILERS:
                expected_params.append(sw_version)

        self.assertEqual(
            len(self.param_val_matrix.keys()),
            len(expected_params),
            f"\nparam_val_matrix: {self.param_val_matrix.keys()}\n"
            f"expected_params: {expected_params}]",
        )

    def test_number_host_device_compiler(self):
        extended_versions = copy.deepcopy(VERSIONS)
        # filter clang-cuda 13 and older because the pair-wise generator cannot filter it out
        # afterwards
        extended_versions[CLANG_CUDA] = extended_versions[CLANG]

        number_of_host_compilers = 0
        for compiler in COMPILERS:
            number_of_host_compilers += len(extended_versions[compiler])

        # NVCC is only as device compiler added
        number_of_device_compilers = number_of_host_compilers

        self.assertEqual(len(self.param_val_matrix[HOST_COMPILER]), number_of_host_compilers)
        self.assertEqual(len(self.param_val_matrix[DEVICE_COMPILER]), number_of_device_compilers)

    def test_number_of_backends(self):
        for backend in BACKENDS:
            if backend == ALPAKA_ACC_GPU_CUDA_ENABLE:
                # all nvcc versions are also CUDA SDK versions
                # add 1 for disable the backend
                number_of_cuda_backends = len(VERSIONS[NVCC]) + 1
                self.assertEqual(
                    len(self.param_val_matrix[backend]),
                    number_of_cuda_backends,
                    f"\nBackend: {backend}",
                )
            else:
                # normal backends are enabled or not
                self.assertEqual(len(self.param_val_matrix[backend]), 2, f"\nBackend: {backend}")

    def test_number_other_software_versions(self):
        # all software versions and similar which are not compiler versions adds a parameter-value
        # for each version
        for sw_name, sw_versions in VERSIONS.items():
            if sw_name not in COMPILERS:
                self.assertEqual(len(self.param_val_matrix[sw_name]), len(sw_versions))
