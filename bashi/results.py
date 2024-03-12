"""Create list of expected parameter-value-pairs respecting bashi filter rules"""

import copy
from typing import List, Optional
from typeguard import typechecked
from packaging.specifiers import SpecifierSet
from bashi.types import ParameterValue, ParameterValuePair, ParameterValueMatrix
from bashi.utils import (
    get_expected_parameter_value_pairs,
    remove_parameter_value_pairs,
)

from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import (
    COMPILERS,
    NVCC_GCC_MAX_VERSION,
    NVCC_CLANG_MAX_VERSION,
    NvccHostSupport,
)


@typechecked
def get_expected_bashi_parameter_value_pairs(
    parameter_matrix: ParameterValueMatrix,
) -> List[ParameterValuePair]:
    """Takes parameter-value-matrix and creates a list of all expected parameter-values-pairs
    allowed by the bashi library. First it generates a complete list of parameter-value-pairs and
    then it removes all pairs that are not allowed by filter rules.

    Args:
        parameter_matrix (ParameterValueMatrix): matrix of parameter values

    Returns:
        List[ParameterValuePair]: list of all parameter-value-pairs supported by bashi
    """
    local_parameter_matrix = copy.deepcopy(parameter_matrix)

    def remove_host_compiler_nvcc(param_val: ParameterValue) -> bool:
        if param_val.name == NVCC:
            return False
        return True

    # remove nvcc as host compiler
    local_parameter_matrix[HOST_COMPILER] = list(
        filter(remove_host_compiler_nvcc, local_parameter_matrix[HOST_COMPILER])
    )

    # remove clang-cuda 13 and older
    def remove_unsupported_clang_cuda_version(param_val: ParameterValue) -> bool:
        if param_val.name == CLANG_CUDA and param_val.version < packaging.version.parse("14"):
            return False
        return True

    local_parameter_matrix[HOST_COMPILER] = list(
        filter(remove_unsupported_clang_cuda_version, local_parameter_matrix[HOST_COMPILER])
    )
    local_parameter_matrix[DEVICE_COMPILER] = list(
        filter(remove_unsupported_clang_cuda_version, local_parameter_matrix[DEVICE_COMPILER])
    )

    param_val_pair_list = get_expected_parameter_value_pairs(local_parameter_matrix)

    _remove_unsupported_nvcc_host_compiler(param_val_pair_list)
    _remove_different_compiler_names(param_val_pair_list)
    _remove_different_compiler_versions(param_val_pair_list)
    _remove_nvcc_unsupported_gcc_versions(param_val_pair_list)
    _remove_nvcc_unsupported_clang_versions(param_val_pair_list)
    _remove_specific_nvcc_clang_combinations(param_val_pair_list)

    return param_val_pair_list


def _remove_unsupported_nvcc_host_compiler(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all combinations where nvcc is device compiler and the host compiler is not gcc or
    clang.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    for compiler_name in set(COMPILERS) - set([GCC, CLANG, NVCC]):
        remove_parameter_value_pairs(
            parameter_value_pairs,
            parameter1=HOST_COMPILER,
            value_name1=compiler_name,
            parameter2=DEVICE_COMPILER,
            value_name2=NVCC,
        )


def _remove_different_compiler_names(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all combinations, where host and device compiler name are different except the device
    compiler name is nvcc.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    # remove all combinations, where host and device compiler name are different except the device
    # compiler name is nvcc
    for host_compiler_name in set(COMPILERS) - set([NVCC]):
        for device_compiler_name in set(COMPILERS) - set([NVCC]):
            if host_compiler_name != device_compiler_name:
                remove_parameter_value_pairs(
                    parameter_value_pairs,
                    parameter1=HOST_COMPILER,
                    value_name1=host_compiler_name,
                    parameter2=DEVICE_COMPILER,
                    value_name2=device_compiler_name,
                )


def _remove_different_compiler_versions(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all combinations, where host and device compiler name are equal and versions are
    different except the compiler name is nvcc.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """

    def filter_function(param_val_pair: ParameterValuePair) -> bool:
        param_names = (HOST_COMPILER, DEVICE_COMPILER)
        compiler_names = set(COMPILERS) - set([NVCC])

        if (
            param_val_pair.first.parameter in param_names
            and param_val_pair.second.parameter in param_names
            and param_val_pair.first.parameterValue.name
            == param_val_pair.second.parameterValue.name
            and param_val_pair.first.parameterValue.name in compiler_names
            and param_val_pair.first.parameterValue.version
            != param_val_pair.second.parameterValue.version
        ):
            return False

        return True

    parameter_value_pairs[:] = list(filter(filter_function, parameter_value_pairs))


def _remove_nvcc_unsupported_gcc_versions(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all gcc version, which are to new for a specific nvcc version.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    _remove_unsupported_nvcc_host_compiler_versions(
        parameter_value_pairs, GCC, NVCC_GCC_MAX_VERSION
    )


def _remove_nvcc_unsupported_clang_versions(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all clang version, which are to new for a specific nvcc version.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    _remove_unsupported_nvcc_host_compiler_versions(
        parameter_value_pairs, CLANG, NVCC_CLANG_MAX_VERSION
    )


def _remove_unsupported_nvcc_host_compiler_versions(
    parameter_value_pairs: List[ParameterValuePair],
    host_compiler_name: str,
    support_list: List[NvccHostSupport],
):
    # pylint: disable=too-few-public-methods
    class _FilterFunctor:
        def __init__(
            self,
            host_compiler_name: str,
            inklusiv_min_version: Optional[NvccHostSupport] = None,
            exklusiv_max_version: Optional[NvccHostSupport] = None,
        ) -> None:
            self.host_compiler_name = host_compiler_name
            if inklusiv_min_version and exklusiv_max_version:
                if inklusiv_min_version.host == exklusiv_max_version.host:
                    self.host_specifier_set = SpecifierSet(f">{exklusiv_max_version.host}")
                else:
                    self.host_specifier_set = SpecifierSet(f">={exklusiv_max_version.host}")
                self.nvcc_specifier_set = SpecifierSet(
                    f">={inklusiv_min_version.nvcc},<{exklusiv_max_version.nvcc}"
                )
            elif inklusiv_min_version:
                self.host_specifier_set = SpecifierSet(f">{inklusiv_min_version.host}")
                self.nvcc_specifier_set = SpecifierSet(f"=={inklusiv_min_version.nvcc}")
            elif exklusiv_max_version:
                self.host_specifier_set = SpecifierSet(f">{exklusiv_max_version.host}")
                self.nvcc_specifier_set = SpecifierSet(f"=={exklusiv_max_version.nvcc}")
            else:
                raise RuntimeError(
                    "at least inklusiv_min_version or exklusiv_max_version needs to be set"
                )

        def __call__(self, param_val_pair: ParameterValuePair) -> bool:
            if (
                param_val_pair.first.parameter == HOST_COMPILER
                and param_val_pair.second.parameter == DEVICE_COMPILER
            ):
                host_param_val = param_val_pair.first.parameterValue
                nvcc_param_val = param_val_pair.second.parameterValue
            elif (
                param_val_pair.first.parameter == HOST_COMPILER
                and param_val_pair.second.parameter == DEVICE_COMPILER
            ):
                host_param_val = param_val_pair.second.parameterValue
                nvcc_param_val = param_val_pair.first.parameterValue
            else:
                return True

            if host_param_val.name == self.host_compiler_name and nvcc_param_val.name == NVCC:
                if (
                    nvcc_param_val.version in self.nvcc_specifier_set
                    and host_param_val.version in self.host_specifier_set
                ):
                    return False

            return True

    oldest_nvcc_first = sorted(support_list)

    for index in range(len(oldest_nvcc_first) - 1):
        filter_function = _FilterFunctor(
            host_compiler_name, oldest_nvcc_first[index], oldest_nvcc_first[index + 1]
        )

        parameter_value_pairs[:] = filter(filter_function, parameter_value_pairs)

    # lower bound
    parameter_value_pairs[:] = filter(
        _FilterFunctor(host_compiler_name, inklusiv_min_version=oldest_nvcc_first[0]),
        parameter_value_pairs,
    )
    # upper bound
    parameter_value_pairs[:] = filter(
        _FilterFunctor(host_compiler_name, exklusiv_max_version=oldest_nvcc_first[-1]),
        parameter_value_pairs,
    )


def _remove_specific_nvcc_clang_combinations(parameter_value_pairs: List[ParameterValuePair]):
    """Remove all pairs, where clang is host-compiler for nvcc 11.3, 11.4 and 11.5 as device
    compiler.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): parameter-value-pair list
    """
    remove_parameter_value_pairs(
        parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=CLANG,
        value_version1=ANY_VERSION,
        parameter2=DEVICE_COMPILER,
        value_name2=NVCC,
        value_version2="!=11.3,!=11.4,!=11.5",
    )