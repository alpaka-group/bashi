"""Help functions to remove parameter-value-pairs with SDK relation."""

from typing import List, Dict, Callable
from packaging.specifiers import SpecifierSet
from bashi.types import Parameter, ValueName, ParameterValueSingle, ParameterValuePair
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.versions import UbuntuSDKMinMax


def remove_unsupported_sdk_ubuntu_combinations(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    sdk_parameter: Parameter,
    sdk_value_name: ValueName,
    ubuntu_sdk_version_range: List[UbuntuSDKMinMax] | Dict[ValueVersion, SpecifierSet],
):
    """Remove all pairs where SDK does not support a specific Ubuntu version

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
        removed_parameter_value_pairs (List[ParameterValuePair): list with removed
            parameter-value-pairs
        sdk_parameter (Parameter): Parameter of the SDK
        sdk_value_name (ValueName): Name of the SDK
        ubuntu_sdk_version_range (List[UbuntuSDKMinMax] | Dict[ValueVersion, SpecifierSet]): Version
            range which specified, which combination is valid.
    """
    ub_sdk_ranges: Dict[ValueVersion, SpecifierSet] = {}
    if isinstance(ubuntu_sdk_version_range, (List, UbuntuSDKMinMax)):
        for ub_sdk in ubuntu_sdk_version_range:
            ub_sdk_ranges[ub_sdk.ubuntu] = ub_sdk.sdk_range
    else:
        ub_sdk_ranges = ubuntu_sdk_version_range

    tmp_parameter_value_pairs: List[ParameterValuePair] = []

    def is_remove(param_val1: ParameterValueSingle, param_val2: ParameterValueSingle) -> bool:
        if (
            param_val1.parameter == UBUNTU
            and param_val2.parameter == sdk_parameter
            and param_val2.parameterValue.name == sdk_value_name
            # special case: backend is disabled
            # assumption: no Compiler has the version number 0.0.0
            and param_val2.parameterValue.version != OFF_VER
        ):
            if param_val1.parameterValue.version not in ub_sdk_ranges:
                return True

            if (
                param_val2.parameterValue.version
                not in ub_sdk_ranges[param_val1.parameterValue.version]
            ):
                return True

        return False

    for param_val in parameter_value_pairs:
        if is_remove(param_val.first, param_val.second) or is_remove(
            param_val.second, param_val.first
        ):
            removed_parameter_value_pairs.append(param_val)
        else:
            tmp_parameter_value_pairs.append(param_val)

    parameter_value_pairs[:] = tmp_parameter_value_pairs


def remove_runtime_not_available_ubuntu_versions(
    parameter_value_pairs: List[ParameterValuePair],
    removed_parameter_value_pairs: List[ParameterValuePair],
    runtime_info: Callable[..., bool],
    backend_name: Parameter,
):
    """Remove all Ubuntu Backend combinations, which are not available because of the input
    parameter-value-matrix.

    Args:
        parameter_value_pairs (List[ParameterValuePair]): List of parameter-value pairs.
        removed_parameter_value_pairs (List[ParameterValuePair): list with removed
        parameter-value-pairs
        runtime_info (Callable[..., bool]): Functor which decides if a Ubuntu version is available.
        backend_name (Parameter): Name of the backend
    """
    tmp_parameter_value_pairs: List[ParameterValuePair] = []

    def is_remove(param_val1: ParameterValueSingle, param_val2: ParameterValueSingle) -> bool:
        if (
            param_val1.parameter == backend_name
            and param_val1.parameterValue.version != OFF_VER
            and param_val2.parameter == UBUNTU
            and not runtime_info(param_val2.parameterValue.version)
        ):
            return True

        return False

    for param_val in parameter_value_pairs:
        if is_remove(param_val.first, param_val.second) or is_remove(
            param_val.second, param_val.first
        ):
            removed_parameter_value_pairs.append(param_val)
        else:
            tmp_parameter_value_pairs.append(param_val)

    parameter_value_pairs[:] = tmp_parameter_value_pairs
