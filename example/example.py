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
import sys
import packaging.version as pkv
from bashi.generator import generate_combination_list
from bashi.utils import (
    check_parameter_value_pair_in_combination_list,
    remove_parameter_value_pairs,
)
from bashi.results import get_expected_bashi_parameter_value_pairs
from bashi.types import (
    ParameterValue,
    ParameterValuePair,
    ParameterValueTuple,
    ParameterValueMatrix,
    Combination,
    CombinationList,
)
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import (
    get_parameter_value_matrix,
    VERSIONS,
    NVCC_GCC_MAX_VERSION,
    NVCC_CLANG_MAX_VERSION,
)


# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
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
    cpu_backends = set(BACKENDS) - gpu_backends
    gpu_compilers = set([NVCC, CLANG_CUDA, HIPCC, ICPX])
    cpu_compilers = set(COMPILERS) - gpu_compilers

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
                    remove_parameter_value_pairs(
                        expected_param_val_tuple,
                        parameter1=gpu_backend,
                        value_name1=gpu_backend,
                        value_version1=gpu_version,
                        parameter2=other_backend,
                        value_name2=other_backend,
                        value_version2=other_backend_version,
                    )

    # remove all pairs, which contains two cpu backends and on of the backends is enabled and the
    # other is disabled
    for cpu_backend in cpu_backends:
        for other_cpu_backend in cpu_backends:
            if cpu_backend != other_cpu_backend:
                remove_parameter_value_pairs(
                    expected_param_val_tuple,
                    parameter1=cpu_backend,
                    value_name1=cpu_backend,
                    value_version1=ON,
                    parameter2=other_cpu_backend,
                    value_name2=other_cpu_backend,
                    value_version2=OFF,
                )

    # if gpu backend is on, remove all combination with enabled cpu backend
    for gpu_compiler in gpu_compilers:
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            for cpu_backend in cpu_backends:
                remove_parameter_value_pairs(
                    expected_param_val_tuple,
                    parameter1=compiler_type,
                    value_name1=gpu_compiler,
                    value_version1=ANY_VERSION,
                    parameter2=cpu_backend,
                    value_name2=cpu_backend,
                    value_version2=ON,
                )

    # remove all combinations with an enabled and disabled cpu backend
    for cpu_compiler in cpu_compilers:
        for cpu_backend in cpu_backends:
            remove_parameter_value_pairs(
                expected_param_val_tuple,
                parameter1=DEVICE_COMPILER,
                value_name1=cpu_compiler,
                value_version1=ANY_VERSION,
                parameter2=cpu_backend,
                value_name2=cpu_backend,
                value_version2=OFF,
            )

    # nvcc does not support all gcc and clang versions
    # therefore there are some gcc and clang host compiler versions which can only works with
    # enabled cpu backends
    max_supported_nvcc_gcc_version = max(comb.host for comb in NVCC_GCC_MAX_VERSION).major
    max_supported_nvcc_clang_version = max(comb.host for comb in NVCC_CLANG_MAX_VERSION).major
    for cpu_backend in cpu_backends:
        remove_parameter_value_pairs(
            expected_param_val_tuple,
            parameter1=HOST_COMPILER,
            value_name1=GCC,
            value_version1=f"<={max_supported_nvcc_gcc_version}",
            parameter2=cpu_backend,
            value_name2=cpu_backend,
            value_version2=OFF,
        )
        remove_parameter_value_pairs(
            expected_param_val_tuple,
            parameter1=HOST_COMPILER,
            value_name1=CLANG,
            value_version1=f"<={max_supported_nvcc_clang_version}",
            parameter2=cpu_backend,
            value_name2=cpu_backend,
            value_version2=OFF,
        )

    def all_cpu_backends_are(expected_state: pkv.Version, combination: Combination) -> bool:
        """Check if all cpu backends ether enabled or disabled

        Args:
            expected_state (pkv.Version): Ether ON_VER or OFF_VER
            combination (Combination): combination to check

        Returns:
            bool: True if all cpu backends have the expected state
        """
        cpu_backends = set(BACKENDS) - set(
            [ALPAKA_ACC_GPU_CUDA_ENABLE, ALPAKA_ACC_GPU_HIP_ENABLE, ALPAKA_ACC_SYCL_ENABLE]
        )

        def version_to_string(version: pkv.Version) -> str:
            if version == ON_VER:
                return "ON"
            if version == OFF_VER:
                return "OFF"
            return "unknown"

        all_right = True
        error_msg = f"ERROR, following combination is wrong:\n  {combination}\n"

        for cpu_backend in cpu_backends:
            if cpu_backend in combination and combination[cpu_backend].version != expected_state:
                error_msg += (
                    f"{cpu_backend} is {version_to_string(combination[cpu_backend].version)} "
                    f"-> expected was {version_to_string(expected_state)}\n"
                )
                all_right = False

        if not all_right:
            print(error_msg)

        return all_right

    all_right = True

    # this loop checks depending on the device compiler the state of the backends
    #  - if the device compiler is a cpu compiler: all cpu backends needs to be enabled, all gpu
    #    backends needs to be disabled
    #  - if the device compiler is a gpu compiler: all cpu backends needs to be disabled, the
    #    related gpu backend needs to be enabled and all other gpu backends needs to be disabled
    # pylint: disable=too-many-nested-blocks
    for comb in combination_list:
        if comb[DEVICE_COMPILER].name in (GCC, CLANG):
            all_right = all_cpu_backends_are(ON_VER, comb) and all_right
            if not (
                comb[ALPAKA_ACC_GPU_CUDA_ENABLE].version == OFF_VER
                and comb[ALPAKA_ACC_GPU_HIP_ENABLE].version == OFF_VER
                and comb[ALPAKA_ACC_SYCL_ENABLE].version == OFF_VER
            ):
                print(f"ERROR: gpu backend is enabled for cpu device compiler:\n  {comb}\n")
        else:
            all_right = all_cpu_backends_are(OFF_VER, comb) and all_right
            for device_compiler, backend_on in (
                (NVCC, ALPAKA_ACC_GPU_CUDA_ENABLE),
                (CLANG_CUDA, ALPAKA_ACC_GPU_CUDA_ENABLE),
                (HIPCC, ALPAKA_ACC_GPU_HIP_ENABLE),
                (ICPX, ALPAKA_ACC_SYCL_ENABLE),
            ):
                if comb[DEVICE_COMPILER].name == device_compiler:
                    if comb[backend_on].version == OFF_VER:
                        print(
                            f"ERROR: {backend_on} needs to be enabled for device compiler "
                            f"{device_compiler}:  \n {comb}\n"
                        )
                        all_right = False
                    for gpu_backend in gpu_backends:
                        if gpu_backend != backend_on and comb[gpu_backend].version != OFF_VER:
                            print(
                                f"ERROR: {backend_on} needs to be disabled for device compiler "
                                f"{device_compiler}:  \n {comb}\n"
                            )
                            all_right = False

    return (
        check_parameter_value_pair_in_combination_list(combination_list, expected_param_val_tuple)
        and all_right
    )


# pylint: disable=too-many-return-statements
def custom_filter(row: ParameterValueTuple) -> bool:
    """Filter function defined by the user. In this case, remove some backend combinations, see
    module documentation.

    Args:
        row (ParameterValueTuple): parameter-value-tuple

    Returns:
        bool: True if the tuple is valid
    """
    gpu_compilers = set([NVCC, CLANG_CUDA, HIPCC, ICPX])
    cpu_compilers = set(COMPILERS) - gpu_compilers

    gpu_backends = set(
        [
            ALPAKA_ACC_GPU_CUDA_ENABLE,
            ALPAKA_ACC_GPU_HIP_ENABLE,
            ALPAKA_ACC_SYCL_ENABLE,
        ]
    )
    cpu_backends = set(BACKENDS) - gpu_backends

    # cpu backend cannot be enabled if gpu compiler is used
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        if compiler_type in row and row[compiler_type].name in gpu_compilers:
            for cpu_backend in cpu_backends:
                if cpu_backend in row and row[cpu_backend].version == ON_VER:
                    return False

    if DEVICE_COMPILER in row:
        for cpu_compiler in cpu_compilers:
            if row[DEVICE_COMPILER].name == cpu_compiler:
                # if the device compiler is a cpu compiler, all cpu backends needs to be enabled
                for cpu_backend in cpu_backends:
                    if cpu_backend in row and row[cpu_backend].version == OFF_VER:
                        return False
                # if the device compiler is a cpu compiler, all gpu backends needs to be disabled
                for gpu_backend in gpu_backends:
                    if gpu_backend in row and row[gpu_backend].version != OFF_VER:
                        return False

    for cpu_backend in cpu_backends:
        # if a single cpu backend is enabled all other cpu backends needs also to be enabled
        if cpu_backend in row and row[cpu_backend].version == ON_VER:
            for cpu_backend2 in cpu_backends:
                if cpu_backend != cpu_backend2:
                    if cpu_backend2 in row and row[cpu_backend2].version == OFF_VER:
                        return False
            # if a single cpu backend is enabled all other gpu backends needs to be disabled
            for gpu_backend in gpu_backends:
                if gpu_backend in row and row[gpu_backend].version != OFF_VER:
                    return False

    # if nvcc does not support a gcc/clang version, the gcc/clang compiler can be only used as cpu
    # compiler
    for compiler_name, max_supported_version in (
        (GCC, max(comb.host for comb in NVCC_GCC_MAX_VERSION)),
        (CLANG, max(comb.host for comb in NVCC_CLANG_MAX_VERSION)),
    ):
        if (
            HOST_COMPILER in row
            and row[HOST_COMPILER].name == compiler_name
            and row[HOST_COMPILER].version > pkv.parse(str(max_supported_version))
        ):
            for cpu_backend in cpu_backends:
                if cpu_backend in row and row[cpu_backend].version == OFF_VER:
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
    # append project specific parameter-values
    param_matrix["SoftwareA"] = [
        ParameterValue("SoftwareA", ValueVersion("1.0")),
        ParameterValue("SoftwareA", ValueVersion("2.0")),
        ParameterValue("SoftwareA", ValueVersion("2.1")),
    ]

    comb_list: CombinationList = generate_combination_list(
        parameter_value_matrix=param_matrix, custom_filter=custom_filter
    )

    create_yaml(comb_list)
    print(f"number of combinations: {len(comb_list)}")

    print("verify combination-list")
    if verify(comb_list, param_matrix):
        print("verification passed")
        sys.exit(0)

    print("verification failed")
    sys.exit(1)
