# Remove unexpected parameter-values-pairs

The `get_expected_parameter_value_pairs()` function generates a list of all possible parameter-value-pairs. The functions `check_parameter_value_pair_in_combination_list()` and `check_unexpected_parameter_value_pair_in_combination_list` search for expected and unexpected parameter-value-pairs in a combination list. Unexpected parameter-value-pairs exist because the filter rules do not allow all possible combinations of parameter-values. The following functions help to remove unexpected parameter-value-pairs from a list of parameter-value-pairs.

## remove_parameter_value_pairs()

The `remove_parameter_value_pairs()` function searches for specific parameters, value-names and value-versions in each parameter value of a parameter-value-pair.

```python
# remove all pairs, which contains host compiler nvcc
remove_parameter_value_pairs(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=NVCC,
    )

# remove all pairs, which contains the host compiler GCC 12
remove_parameter_value_pairs(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=GCC,
        value_version1=12,
    )

# remove the pair with device compiler Clang 17 and CUDA Backend version 12.5
remove_parameter_value_pairs(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=DEVICE_COMPILER,
        value_name1=Clang,
        value_version1=17,
        parameter2=ALPAKA_ACC_CUDA_ENABLE,
        value_name2=ALPAKA_ACC_CUDA_ENABLE,
        value_version2=12.5,
    )
```

## remove_parameter_value_pairs_ranges()

The `remove_parameter_value_pairs_ranges()` function also removes all parameter-value-pairs from a list, but for specific versions ranges. The matching of parameter and value-names works in the same way as with `remove_parameter_value_pairs()`.

The version range to be removed is defined by a minimum and a maximum version. By default, both ends of the version range are open. The version range is restricted by the arguments `value_min_versionX` and `value_max_versionX`.

The following examples show how to define specific ranges to be removed. For the example, we assume that we have major versions from 1 to 9.

```python
# all major version before removed: [1,2,3,4,5,6,7,8,9]

# all ranges are open, therefore remove all versions
remove_parameter_value_pairs_range(
    # ...
) # -> output []

# minimum version to be remove is 3
remove_parameter_value_pairs_range(
    # ...
    value_min_version1=3
) # -> output [1,2]

# minimum version to be remove is 3
# 3 is excluded from the range of version to be removed
remove_parameter_value_pairs_range(
    # ...
    value_min_version1=3,
    value_min_version1_inclusive=False
) # -> output [1,2,3]

# maximum version to be remove is 6
remove_parameter_value_pairs_range(
    # ...
    value_max_version1=6
) # -> output [7,8,9]

# maximum version to be remove is 6
# 6 is excluded from the range of version to be removed
remove_parameter_value_pairs_range(
    # ...
    value_max_version1=6,
    value_max_version1_inclusive=False
) # -> output [6,7,8,9]

# remove all version between 4 and 8
remove_parameter_value_pairs_range(
    # ...
    value_min_version1=4,
    value_max_version1=8,
) # -> output [1,2,3,9]

# remove all version between 4 and 8
# exclude borders
remove_parameter_value_pairs_range(
    # ...
    value_min_version1=4,
    value_min_version1_inclusive=False
    value_max_version1=8,
    value_max_version1_inclusive=False
) # -> output [1,2,3,4,8,9]

# remove only version 5
remove_parameter_value_pairs_range(
    # ...
    value_min_version1=5,
    value_max_version1=5,
) # -> output [1,2,3,4,6,7,8,9]
```

### back-end versions

All back-ends in `bashi` except the CUDA back-end have the states `OFF` or `ON`, which are internally represented by the versions 0.0.0 and 1.0.0. Therefore, ranges can be defined to remove all enabled or disabled back-ends, regardless of whether it is a CUDA or non-CUDA back-end.

```python
# remove all enabled back-ends
# all enabled backends has a version higher than 0.0.0
remove_parameter_value_pairs_range(
    # ...
    value_min_version1=OFF,
    value_min_version1_inclusive=False
)

# use remove_parameter_value_pairs() to remove the disabled back-ends (single version)
remove_parameter_value_pairs(
    # ...
    value_version1=OFF,
)
```
