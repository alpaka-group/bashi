"""The example shows how the bashi library can be used. The example does the following things:

1. generate a parameter-value-matrix with all software versions supported by bashi
2. generate a combination-list
  - the generator uses the bashi filter rules and a custom filter
  - The custom filter filters the backend configurations
    - either all CPU backends and no GPU backend are activated
    - or a single gpu backend is enabled and all other backends are disabled
3. check whether all expected parameter-value pairs are contained in the combination-list
  - all pairs that are prohibited by the user-defined filter are removed from the list of 
    expected parameter-value-pairs
4. generate a job.yaml from the combination-list
"""

from typing import List
import os
from bashi.generator import generate_combination_list
from bashi.utils import (
    get_expected_bashi_parameter_value_pairs,
    check_parameter_value_pair_in_combination_list,
    remove_parameter_value_pair,
    create_parameter_value_pair,
)
from bashi.types import (
    ParameterValuePair,
    ParameterValueTuple,
    ParameterValueMatrix,
    CombinationList,
)
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import get_parameter_value_matrix, VERSIONS


# pylint: disable=too-many-branches
def verify(combination_list: CombinationList, param_value_matrix: ParameterValueMatrix) -> bool:
    """Check if all expected parameter-value-pairs exists in the combination-list.

    Args:
        combination_list (CombinationList): The generated combination list.
        param_value_matrix (ParameterValueMatrix): The expected parameter-values-pairs are generated
            from the parameter-value-list.

    Returns:
        bool: True if it found all pairs
    """
    expected_param_val_tuple: List[ParameterValuePair] = get_expected_bashi_parameter_value_pairs(
        param_value_matrix
    )

    gpu_backends = set(
        [
            ALPAKA_ACC_GPU_CUDA_ENABLE,
            ALPAKA_ACC_GPU_HIP_ENABLE,
            ALPAKA_ACC_SYCL_ENABLE,
        ]
    )

    # if one of the GPU backend is enabled, all other backends needs to be disabled
    # special case CUDA backend: instead it has the version on or off, it has off or a version
    # number
    for gpu_backend in gpu_backends:
        if gpu_backend == ALPAKA_ACC_GPU_CUDA_ENABLE:
            gpu_versions = VERSIONS[NVCC]
        else:
            gpu_versions = [ON]
        for gpu_version in gpu_versions:
            for other_backend in set(BACKENDS) - set([gpu_backend]):
                if other_backend == ALPAKA_ACC_GPU_CUDA_ENABLE:
                    other_backend_versions = VERSIONS[NVCC]
                else:
                    other_backend_versions = [ON]

                for other_backend_version in other_backend_versions:

                    remove_parameter_value_pair(
                        to_remove=create_parameter_value_pair(
                            gpu_backend,
                            gpu_backend,
                            gpu_version,
                            other_backend,
                            other_backend,
                            other_backend_version,
                        ),
                        parameter_value_pairs=expected_param_val_tuple,
                    )
                    remove_parameter_value_pair(
                        to_remove=create_parameter_value_pair(
                            other_backend,
                            other_backend,
                            other_backend_version,
                            gpu_backend,
                            gpu_backend,
                            gpu_version,
                        ),
                        parameter_value_pairs=expected_param_val_tuple,
                    )

    cpu_backends = set(BACKENDS) - gpu_backends
    # remove all pairs, which contains two cpu backends and on of the backends is enabled and the
    # other is disabled
    for cpu_backend in cpu_backends:
        for other_cpu_backend in cpu_backends:
            if cpu_backend != other_cpu_backend:
                remove_parameter_value_pair(
                    to_remove=create_parameter_value_pair(
                        cpu_backend,
                        cpu_backend,
                        ON_VER,
                        other_cpu_backend,
                        other_cpu_backend,
                        OFF_VER,
                    ),
                    parameter_value_pairs=expected_param_val_tuple,
                )
                remove_parameter_value_pair(
                    to_remove=create_parameter_value_pair(
                        other_cpu_backend,
                        other_cpu_backend,
                        OFF_VER,
                        cpu_backend,
                        cpu_backend,
                        ON_VER,
                    ),
                    parameter_value_pairs=expected_param_val_tuple,
                )

    return check_parameter_value_pair_in_combination_list(
        combination_list, expected_param_val_tuple
    )


def custom_filter(row: ParameterValueTuple) -> bool:
    """Filter function defined by the user. In this case, remove some backend combinations, see
    module documentation.

    Args:
        row (ParameterValueTuple): parameter-value-tuple

    Returns:
        bool: True if the tuple is valid
    """
    gpu_backends = set(
        [
            ALPAKA_ACC_GPU_CUDA_ENABLE,
            ALPAKA_ACC_GPU_HIP_ENABLE,
            ALPAKA_ACC_SYCL_ENABLE,
        ]
    )
    for single_gpu_backend in gpu_backends:
        if single_gpu_backend in row and row[single_gpu_backend].version != OFF_VER:
            for backend in BACKENDS:
                if backend != single_gpu_backend:
                    if backend in row and row[backend].version != OFF_VER:
                        return False

    cpu_backends = set(BACKENDS) - gpu_backends
    for cpu_backend in cpu_backends:
        if cpu_backend in row and row[cpu_backend].version == ON_VER:
            # all other cpu backends needs to be enabled
            for other_cpu_backend in cpu_backends - set(cpu_backend):
                if other_cpu_backend in row and row[other_cpu_backend].version == OFF_VER:
                    return False
            # all other gpu backends needs to be disabled
            for gpu_backend in gpu_backends:
                if gpu_backend in row and row[gpu_backend].version != OFF_VER:
                    return False

    return True


def create_yaml(combination_list: CombinationList):
    """Create an example GitLab CI job yaml from the combination-list and write it to a file.
    Normally, a yaml library should be used. But for a small example it is better to create the yaml
    file by hand instead of adding a dependency.

    Args:
        combination_list (CombinationList): combination-list
    """
    job_yaml = ""
    for job_num, comb in enumerate(combination_list):
        job_yaml += f"ci_job_{job_num}:\n"
        job_yaml += "  variables:\n"
        for param, param_val in comb.items():
            val_name, val_version = param_val
            if param == HOST_COMPILER:
                job_yaml += f"    - HOST_COMPILER_NAME: {val_name}\n"
                job_yaml += f"    - HOST_COMPILER_VERSION: {val_version}\n"
            elif param == DEVICE_COMPILER:
                job_yaml += f"    - DEVICE_COMPILER_NAME: {val_name}\n"
                job_yaml += f"    - DEVICE_COMPILER_VERSION: {val_version}\n"
            elif param in BACKENDS:
                if val_version == ON_VER:
                    job_yaml += f"    - {val_name.upper()}: ON\n"
                elif val_version == OFF_VER:
                    job_yaml += f"    - {val_name.upper()}: OFF\n"
                else:
                    job_yaml += f"    - {val_name.upper()}: {val_version}\n"
            else:
                job_yaml += f"    - {val_name.upper()}: {val_version}\n"
        job_yaml += "  script:\n"
        job_yaml += "    - ./run_tests.sh\n"
        job_yaml += "  tags:\n"
        if comb[ALPAKA_ACC_SYCL_ENABLE].version == ON_VER:
            job_yaml += "    - intel-gpu-runner\n"
        elif comb[ALPAKA_ACC_GPU_HIP_ENABLE].version == ON_VER:
            job_yaml += "    - amd-gpu-runner\n"
        elif comb[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER:
            job_yaml += "    - nvidia-gpu-runner\n"
        else:
            job_yaml += "    - cpu-runner\n"

        job_yaml += "\n"

    # generate job.yaml always in the same folder where the example.py is located
    job_yaml_path = os.path.join(os.path.dirname(__file__), "job.yaml")
    print(f"write GitLab CI job.yaml to {job_yaml_path}")
    with open(job_yaml_path, "w", encoding="UTF-8") as output:
        output.write(job_yaml)


if __name__ == "__main__":
    param_matrix = get_parameter_value_matrix()

    comb_list: CombinationList = generate_combination_list(
        parameter_value_matrix=param_matrix, custom_filter=custom_filter
    )

    print("verify combination-list")
    verify(comb_list, param_matrix)

    create_yaml(comb_list)
    print(f"number of combinations: {len(comb_list)}")
