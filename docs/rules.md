# Filter Rules

`Bashi` uses filters to disallow invalid `parameter-value-tuple` (blacklisting). Therefore, new rules reduce the number of valid `parameter-value-tuple`. In practice, this means that a Python function returns `true` if a `parameter-value-tuple` is OK and `false` if a `parameter-value-tuple` is not allowed.

```python
# filters all parameter-value-tuples where the host and device compiler do not have the same name
def example_filter(row : ParameterValueTuple) -> bool:
    if (HOST_COMPILER and row
        and DEVICE_COMPILER in row
        and row[HOST_COMPILER].name != row[DEVICE].name):
            return False

    return True
```

The filter rules are divided into the functions `compiler_filter()`, `backend_filter()` and `software_dependency_filter()` for a better overview. The `get_default_filter_chain()` function defines the sequence in which the filter rules are called.

The pair-wise combination algorithm of the `covertable` library defines the input of the filter function. The pair-wise algorithm attempts to generate as few `combinations` as possible. Therefore, the input has some special properties. The input of a filter rule is a `parameter-value-tuple` (partial `combination`) or a `combination`. This means that each input has one or more `parameter`s, each with an associated `parameter-value`. The order of the `parameter`s is random. Since a `parameter-value-tuple` does not have to contain all parameters, a filter rule must first check whether a `parameter` is present in the `parameter-value-tuple`. It can then check for `value-name` and/or `value-version`.

A `parameter-value-tuple` passes through the filter many times, each time with an additional `parameter` or a different `parameter-value` for the last `parameter` in the ordered dictionary. This means that a `parameter-value-tuple` grows until it contains all `parameter` and the combination of all `parameter-values` is valid.

For example, the filter rule is called with the following sequence of `parameter-value-tuple`s [1].

```bash
host=gcc@13 bTBB=OFF # valid combination
host=gcc@13 bTBB=OFF cmake=3.24 # valid combination
host=gcc@13 bTBB=OFF cmake=3.24 ubuntu=20.4 # valid combination
host=gcc@13 bTBB=OFF cmake=3.24 ubuntu=20.4 device=gcc@11 # not valid because if host and device compiler are gcc, they need to have the same version
host=gcc@13 bTBB=OFF cmake=3.24 ubuntu=20.4 device=gcc@12
host=gcc@13 bTBB=OFF cmake=3.24 ubuntu=20.4 device=gcc@13 # valid combination
host=gcc@13 bTBB=OFF cmake=3.24 ubuntu=20.4 device=gcc@13 bSeq=ON # valid combination
```

# Writing a New Rule

Writing a new rule can be complicated because different rules may interact in ways that the author was not aware of. The example was taken from the development of a new rule for the production code. The author has already implemented some rules, such as that a host and a device compiler must have the same name and the same version, except for `nvcc` as the device compiler.

The following `parameter-value-tuple`[1] throws a `covertable.exceptions.InvalidCondition` error, which in practice means that the `parameter-value-tuple` was already invalid before the `parameter` `bHIP`.

```
host=gcc@13 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=OFF bSeq=OFF bSYCL=OFF bHIP=OFF
```

The problem was analyzed as follows. Only the parameters `bCUDA` for the CUDA back-end and `device` for the device compiler are missing. Let's assume the next parameter is `bCUDA`. `bCUDA=OFF` will not work as this means that the device compiler must be `device=gcc@13` and the code must be compiled for the CPU. Therefore at least one of the CPU back-ends must be enabled. So let's use some version for the CUDA back-end and `nvcc` as the device compiler. This sounds reasonable until we check the available CUDA SDK version. For this example, let's assume we are using CUDA 11.0 to CUDA 12.3. The `combination` is not valid with any CUDA version, as CUDA 12.3 is only supported up to `gcc` 12. Therefore, there was no longer a valid combination at this point:

```
host=gcc@13 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=OFF bSeq=OFF
```

In this case, `covertable` throws the error `covertable.exceptions.InvalidCondition: It will never meet the condition`. `covertable` has a bookkeeping algorithm which writes down whether valid `combinations` should still exist. If there are still valid `combinations` left and these cannot be reached, the `covertable.exceptions.InvalidCondition` error occurs.
Unfortunately, the error message does not indicate which `combination` is not reachable. Also, there is no documentation about the bookkeeping algorithm and due to heavy optimizations in the source code of `covertable` it is also not clear how it works. After writing some filter rules, I found two general rules that avoid the `covertable.exceptions.InvalidCondition`:

1. cancel a `parameter-value-tuple` early as possible.
2. write only rules with two parameters.

## Cancel a Parameter-Value-Tuple early as possible

From the example, at the beginning we know this `parameter-value-tuple` was not canceled early as possible:

```
host=gcc@13 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=OFF bSeq=OFF bSYCL=OFF bHIP=OFF
```

The reason is, because there is no `CUDA` version up to `CUDA` 12.3, which support `gcc` 13, the `CUDA` back-end needs to be disabled. This means we can only compile CPU back-ends with `gcc` 13. Another rules says, that at least one CPU back-end needs to be enabled, if we want to compile for the CPU back-end. So it means with the first `parameter-value` know that only valid `combinations` exist, where the GPU back-ends needs to be disabled and at least on CPU back-end needs to be enabled. So the `parameter-value-tuple` needs to be canceled after the last `parameter-value` of a disabled CPU back-end (`bSeq=OFF`) was added. Otherwise the book keeping algorithm thinks, there should be a valid `combination` with host compiler `gcc` 13 and all disabled CPU back-ends. Than it tries all following combinations of the last missing `parameter-values`, when all existing CPU back-ends was set and didn't find any valid combination. In this case, it throws the error `covertable.exceptions.InvalidCondition`.

The reason for this is that there is no CUDA version up to CUDA 12.3 that supports `gcc` 13, so the CUDA back-end must be disabled. This means that we can only compile CPU back-ends with `gcc` 13. Another rule says that at least one CPU back-end must be enabled if we want to compile for the CPU back-end. So this means that with the first `parameter-value` there are only valid `combinations` where the GPU back-ends must be disabled and at least one CPU back-end must be enabled. So the `parameter-value-tuple` must be canceled after the last `parameter-value` of a deactivated CPU back-end (`bSeq=OFF`) has been added. Otherwise, the bookkeeping algorithm thinks that there should be a valid `combination` with the host compiler `gcc` 13 and all disabled CPU back-ends. It then tries all subsequent combinations of the last missing `parameter-values` when all existing CPU back-ends have been set and finds no valid combination. In this case, the error `covertable.exceptions.InvalidCondition` is thrown.

### Debugging Rules

The interaction between the rules makes it difficult to understand why a `covertable.exceptions.InvalidCondition` exception was thrown for a particular `parameter-value-tuple` and where the `parameter-value-tuple` should already be aborted. `bashi` provides some tools to facilitate the analysis. First you should uncomment the function `print_row_nice()` at the beginning of the filter rule of `compiler_filter()`. Then you should add a `print("passed")` at the end of the function `software_dependency_filter()`. With both annotations, `bashi` shows what the last tested `parameter-value-tuple` was and whether it passed the filter:

```bash
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=hipcc@5.0
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=hipcc@5.1
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=hipcc@5.2
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=hipcc@5.3
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=hipcc@5.4
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=hipcc@5.5
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=hipcc@5.6
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=hipcc@5.7
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=hipcc@6.0
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=icpx@2023.1.0
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=icpx@2023.2.0
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=clang-cuda@6
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=clang-cuda@7
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=clang-cuda@8
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=clang-cuda@9
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=clang-cuda@10
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=clang-cuda@11
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=clang-cuda@12
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=clang-cuda@13
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=clang-cuda@14
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=clang-cuda@15
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=clang-cuda@16
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF device=clang-cuda@17
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF
pass
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF
pass
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF
pass
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF
pass
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF
pass
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF
pass
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF
pass
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF
pass
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF
pass
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF
pass
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF
pass
host=clang@11 bCUDA=11.0 bTBB=OFF cmake=3.24 ubuntu=20.4 boost=1.74.0 bOpenMP2block=OFF bThreads=OFF c++=17 bOpenMP2thread=ON bSeq=OFF bSYCL=OFF bHIP=OFF
pass
Traceback (most recent call last):
  File "/home/simeon/projects/bashi/example/example.py", line 215, in <module>
    comb_list: CombinationList = generate_combination_list(
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/simeon/projects/bashi/bashi/generator.py", line 39, in generate_combination_list
    all_pairs: List[Dict[Parameter, ParameterValue]] = make(
                                                       ^^^^^
  File "/home/simeon/.local/miniconda3/envs/bashi-covertable/lib/python3.12/site-packages/covertable/main.py", line 173, in make
    return list(gen)
           ^^^^^^^^^
  File "/home/simeon/.local/miniconda3/envs/bashi-covertable/lib/python3.12/site-packages/covertable/main.py", line 141, in make_async
    row.complement()
  File "/home/simeon/.local/miniconda3/envs/bashi-covertable/lib/python3.12/site-packages/covertable/main.py", line 79, in complement
    raise InvalidCondition(InvalidCondition.message)
covertable.exceptions.InvalidCondition: It will never meet the condition
```

If you cannot directly recognize why a `parameter-value-tuple` was cancel early enough, you can use `bashi-validate` (installed with the `bashi` package) to check why a `parameter-value-tuple` does not pass the filter. Usually it is sufficient to test different `parameter-values` of the last `parameter` or the last missing `parameter`. You can set `print_row_nice(row, bashi_validate=True)` and run your code again. This time the row will be output slightly differently. You can pass the output directly as an argument to `bashi-validate`:

```bash
# code executed again with print_row_nice(row, bashi_validate=True)
--host=clang@11 --bCUDA=11.0 --bTBB=OFF --cmake=3.24 --ubuntu=20.4 --boost=1.74.0 --bOpenMP2block=OFF --bThreads=OFF --c++=17 --bOpenMP2thread=ON --bSeq=OFF --bSYCL=OFF --bHIP=OFF
pass
Traceback (most recent call last):
  File "/home/simeon/projects/bashi/example/example.py", line 215, in <module>
    comb_list: CombinationList = generate_combination_list(
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/simeon/projects/bashi/bashi/generator.py", line 39, in generate_combination_list
    all_pairs: List[Dict[Parameter, ParameterValue]] = make(
                                                       ^^^^^
  File "/home/simeon/.local/miniconda3/envs/bashi-covertable/lib/python3.12/site-packages/covertable/main.py", line 173, in make
    return list(gen)
           ^^^^^^^^^
  File "/home/simeon/.local/miniconda3/envs/bashi-covertable/lib/python3.12/site-packages/covertable/main.py", line 141, in make_async
    row.complement()
  File "/home/simeon/.local/miniconda3/envs/bashi-covertable/lib/python3.12/site-packages/covertable/main.py", line 79, in complement
    raise InvalidCondition(InvalidCondition.message)
covertable.exceptions.InvalidCondition: It will never meet the condition

# bashi-validate said that the last parameter-value-tuple is fine
$ bashi-validate --host=clang@11 --bCUDA=11.0 --bTBB=OFF --cmake=3.24 --ubuntu=20.4 --boost=1.74.0 --bOpenMP2block=OFF --bThreads=OFF --c++=17 --bOpenMP2thread=ON --bSeq=OFF --bSYCL=OFF --bHIP=OFF
compiler_filter() returns True
backend_filter() returns True
software_dependency_filter() returns True

# bCUDA is activated, therefore the device compiler must be nvcc or clang-cuda. clang-cuda cannot be combined with the clang host compiler. Therefore, nvcc is the only possible solution. Add --device=nvcc@11.0 as a rule states that the cuda backend and nvcc must have the same version.
$ bashi-validate --host=clang@11 --bCUDA=11.0 --bTBB=OFF --cmake=3.24 --ubuntu=20.4 --boost=1.74.0 --bOpenMP2block=OFF --bThreads=OFF --c++=17 --bOpenMP2thread=ON --bSeq=OFF --bSYCL=OFF --bHIP=OFF --device=nvcc@11.0
compiler_filter() returns False
  nvcc 11.0 does not support clang 11
backend_filter() returns True
software_dependency_filter() returns True
```

So we realize that the problem is that `nvcc` 11.0 and `clang` 11 do not work together. This means that we have to cancel the `parameter-value-tuple` already at `host=clang@11 bCUDA=11.0` due to a number of rules.

1. if the cuda back-end is enabled, the device compiler can only be `nvcc` and `clang-cuda` -> we check if `device=clang-cuda` is possible
2. if the device compiler is `clang-cuda`, the host compiler must also be `clang-cuda` -> is not possible because the host compiler is `clang`
3. if the `cuda` back-end is activated and `nvcc` is the device compiler, both must have the same version -> we checked this with `bashi-validate` and found another rule
4. certain `nvcc` device compiler versions only support certain `clang` host compiler versions -> Due to the rule that `nvcc` and CUDA back-end must have the same version, we found a new rule. If the CUDA back-end is enabled and `clang` is the host compiler, we need to check if the `clang` version is supported by the `nvcc` version that has the same version as the CUDA back-end. If we implement this rule, the combination is canceled after the second parameter and `covertable` is fine with our filter rule set and generates a `combination-matrix`.


## Write only Rules with two Parameters

Almost all rules prohibit the combination of two or more `parameter-values`. For example, a rule with two `parameter-values` could be that only a certain `CMake 3.22` or newer is available on `Ubuntu 20.04` because `CMake 3.21` and older is not available in the apt repositories.

Most of the rules contains three and more `parameter-values`. For example we want to add the rule if `hipcc` is the compiler, the `hip` back-end needs to be enabled. There are three `parameter-value`s, because each `combination` contains a host and device compiler. There is a existing rule, which says, if the device compiler is not `nvcc`, the host and device compiler needs to have the same name. This means if the first time the `parameter` `HOST_COMPILER` or `DEVICE_COMPILER` appears and has the name `hipcc` or the `parameter` `ALPAKA_ACC_GPU_HIP_ENABLE` appears and is `ON`, only the following combination and it's permutation of all three `parameter`-`parmeter-value` combinations are possible:

Most rules contain three or more `parameter-values`. For example, we want to add the rule if `hipcc` is the compiler, the `hip` back-end must be enabled. There are three `parameter-values` because each `combination` contains a host and a device compiler. There is an existing rule that says that if the device compiler is not `nvcc`, the host and device compiler must have the same name and version. This means that the first time the `parameter` `HOST_COMPILER` or `DEVICE_COMPILER` appears and has the name `hipcc`, the `parameter` `ALPAKA_ACC_GPU_HIP_ENABLE` must be `ON` when it appears to pass the rule. Conversely, if the parameter `ALPAKA_ACC_GPU_HIP_ENABLE` is `ON` and the `parameter` `HOST_COMPILER` or `DEVICE_COMPILER` appears, it must have the `value-name` `hipcc`.

The permutation of all three `parameter`-`parameter-value` combinations results in the following possibilities:

```bash
# x needs to be the same version
host=hipcc@x device=hipcc@x bHip=ON
host=hipcc@x bHip=ON device=hipcc@x
device=hipcc@x host=hipcc@x bHip=ON
device=hipcc@x bHip=ON host=hipcc@x
bHip=ON host=hipcc@x device=hipcc@x
bHip=ON device=hipcc@x host=hipcc@x
```

Because of this permutation and the first rule "Cancel a combination as early as possible", we have to split the filter rule with three `parameter-values` into rules with two `parameter-values`.

The filter rule of three `parameter`s would be:

1. if the host compiler is `hipcc`, the `hip` back-end needs to be enabled
2. if the device compiler is `hipcc` the `hip` back-end needs to be enabled
3. if the `hip` back-end is enabled, the host compiler needs to be `hipcc`
4. if the `hip` back-end is enabled, the device compiler needs to be `hipcc`
5. (if the host compiler is `hipcc`, the device compiler needs to be `hipcc` with the same version) -> only for completes - already implemented
6. (if the device compiler is `hipcc`, the host compiler needs to be `hipcc` with the same version) -> only for completes - already implemented

For the implementation, it can be reduced to 3 rules, as the order of the `parameters` is irrelevant. Since each rule checks whether both `parameters` are present, it does not matter whether the host compiler `parameters` appears first and then the back-end `parameters` or vice versa. Therefore, only rules 1, 2 and 5 need to be implemented.


# Notes

[1] The examples were printed using the auxiliary function `print_row_nice()`. The function prints a `parameter-value-tuple` with shortened `parameter`s and `parameter-name`s to make the output more readable. The following parameters are special:

- `host`: means the `host-compiler`
- `device`: means the `device-compiler`
- `b<Something>`: all parameters that begin with a lowercase `b` followed by a term with a capital letter name a back-end
