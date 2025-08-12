# pylint: disable=missing-docstring
import sys
import os
import unittest
import bashiValidate

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "example"))
from example.validate import main as validator_main


class TestBashiValidate(unittest.TestCase):
    def test_bashi_validate_valid_input(self):
        validator = bashiValidate.Validator(
            args=[
                "--host-compiler",
                "gcc@11",
                "--device-compiler",
                "nvcc@12.4",
                "--alpaka_ACC_GPU_CUDA_ENABLE",
                "12.4",
                "--alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE",
                "ON",
            ],
            silent=True,
        )
        self.assertTrue(validator.validate())

    def test_bashi_validate_invalid_input(self):
        validator = bashiValidate.Validator(
            args=["--host-compiler", "gcc@12", "--device-compiler", "gcc@11"], silent=True
        )
        self.assertFalse(validator.validate())


class TestExampleValidator(unittest.TestCase):
    def test_example_validator_valid_input(self):
        self.assertTrue(validator_main(["--cxx", "20", "--SoftwareA", "2.0"], True))

    def test_example_validator_invalid_input(self):
        self.assertFalse(
            validator_main(
                [
                    "--host-compiler",
                    "gcc@11",
                    "--device-compiler",
                    "nvcc@12.4",
                    "--alpaka_ACC_GPU_CUDA_ENABLE",
                    "12.4",
                    "--alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE",
                    "ON",
                    "--SoftwareA",
                    "2.0",
                ],
                True,
            )
        )
