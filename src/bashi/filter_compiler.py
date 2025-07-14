"""Filter rules basing on host and device compiler names and versions.

All rules implemented in this filter have an identifier that begins with "c" and follows a number.
Examples: c1, c42, c678 ...

These identifiers are used in the test names, for example, to make it clear which test is testing
which rule.
"""

from typing import Dict, Optional, IO, List, Callable
import packaging.version as pkv
from typeguard import typechecked
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValueTuple, Parameter, ValueName
from bashi.versions import (
    CompilerCxxSupport,
    NVCC_GCC_MAX_VERSION,
    NVCC_CLANG_MAX_VERSION,
    CLANG_CUDA_MAX_CUDA_VERSION,
    GCC_CXX_SUPPORT_VERSION,
    CLANG_CXX_SUPPORT_VERSION,
    CLANG_CUDA_CXX_SUPPORT_VERSION,
    NVCC_CXX_SUPPORT_VERSION,
    MAX_CUDA_SDK_CXX_SUPPORT,
    ICPX_CXX_SUPPORT_VERSION,
    HIPCC_CXX_SUPPORT_VERSION,
)
from bashi.filter import FilterBase
from bashi.utils import reason

# uncomment me for debugging
# from bashi.utils import print_row_nice


def _remove_unsupported_compiler_cxx_combination(
    row: ParameterValueTuple,
    compiler_name: ValueName,
    compiler_type: Parameter,
    compiler_cxx_list: List[CompilerCxxSupport],
    output: Optional[IO[str]] = None,
):
    """Return True, if the compiler version C++ standard combination is not supported.

    Attention: Because of performance reasons, does not check if CXX_STANDARD is in the row.

    Args:
        row (ParameterValueTuple): current row with parameter value
        compiler_name (ValueName): name of the compiler
        compiler_type (Parameter): HOST_COMPILER or DEVICE_COMPILER
        compiler_cxx_list (List[CompilerCxxSupport]): list containing which compiler version added
            support for a new C++ standard
        output (Optional[IO[str]], optional): Writes the reason in the io object why the parameter
            value tuple does not pass the filter. If None, no information is provided. The default
            value is None.

    Returns:
        bool: true if not supported
    """
    if compiler_type in row and row[compiler_type].name == compiler_name:
        for compiler_cxx_ver in compiler_cxx_list:
            if row[compiler_type].version >= compiler_cxx_ver.compiler:
                if row[CXX_STANDARD].version > compiler_cxx_ver.cxx:
                    reason(
                        output,
                        f"{compiler_type} {compiler_name} {row[compiler_type].version} does not "
                        f"support C++{row[CXX_STANDARD].version}",
                    )
                    return True
                # break loop otherwise the C++ support of an older GCC version is
                # applied
                break
        # handle case, if GCC version is older, than the oldest defined GCC C++ support
        # entry
        if (
            row[compiler_type].version < compiler_cxx_list[-1].compiler
            and row[CXX_STANDARD].version >= compiler_cxx_list[-1].cxx
        ):
            reason(
                output,
                f"{compiler_type} {compiler_name} {row[compiler_type].version} does not "
                f"support C++{row[CXX_STANDARD].version}",
            )
            return True
    return False


def _get_max_supported_cxx_version_for_cuda_sdk_for_nvcc(
    cuda_sdk_version: pkv.Version,
    nvcc_compiler_cxx_support_list: List[CompilerCxxSupport],
) -> pkv.Version:
    """Maximum support C++ standard for given CUDA SDK version if the nvcc compiler is used.

    Args:
        cuda_sdk_version (pkv.Version): CUDA backend version
        nvcc_compiler_cxx_support_list (List[CompilerCxxSupport]): Only for testing
        purpose. Use NVCC_CXX_SUPPORT_VERSION.

    Returns:
        pkv.Version: C++ version
    """
    nvcc_cxx_support_version_sorted = sorted(nvcc_compiler_cxx_support_list, reverse=True)
    for nvcc_cxx_version in nvcc_cxx_support_version_sorted:
        if cuda_sdk_version >= nvcc_cxx_version.compiler:
            return nvcc_cxx_version.cxx

    return nvcc_cxx_support_version_sorted[-1].cxx


def _get_max_supported_cxx_version_for_cuda_sdk_for_clang_cuda(
    cuda_sdk_version: pkv.Version,
    max_cuda_sdk_cxx_support: List[CompilerCxxSupport],
) -> pkv.Version:
    """Maximum support C++ standard for given CUDA SDK version if the Clang-CUDA compiler is used.

    Args:
        cuda_sdk_version (pkv.Version): CUDA backend version
        max_cuda_sdk_cxx_support (List[CompilerCxxSupport]): Only for testing
        purpose. Use MAX_CUDA_SDK_CXX_SUPPORT.

    Returns:
        pkv.Version: C++ version
    """
    max_cuda_sdk_cxx_support_sorted = sorted(max_cuda_sdk_cxx_support, reverse=True)
    for cuda_sdk_cxx in max_cuda_sdk_cxx_support_sorted:
        if cuda_sdk_version >= cuda_sdk_cxx.compiler:
            return cuda_sdk_cxx.cxx

    # fallback, return oldest C++ standard
    return max_cuda_sdk_cxx_support_sorted[-1].cxx


def _get_max_supported_cxx_version_for_cuda_sdk(
    cuda_sdk_version: pkv.Version,
    nvcc_compiler_cxx_support_list: List[CompilerCxxSupport],
    max_cuda_sdk_cxx_support: List[CompilerCxxSupport],
) -> pkv.Version:
    """Get the maximum possible supported C++ standard for a given CUDA SDK version.

    Args:
        cuda_sdk_version (pkv.Version): CUDA backen version
        nvcc_compiler_cxx_support_list (List[CompilerCxxSupport]): Only for testing
        purpose. Use NVCC_CXX_SUPPORT_VERSION.
                max_cuda_sdk_cxx_support (List[CompilerCxxSupport]): Only for testing
        purpose. Use MAX_CUDA_SDK_CXX_SUPPORT.

    Returns:
        pkv.Version: C++ version
    """
    return max(
        _get_max_supported_cxx_version_for_cuda_sdk_for_nvcc(
            cuda_sdk_version, nvcc_compiler_cxx_support_list
        ),
        _get_max_supported_cxx_version_for_cuda_sdk_for_clang_cuda(
            cuda_sdk_version, max_cuda_sdk_cxx_support
        ),
    )


# pylint: disable=too-many-branches
# pylint: disable=too-many-return-statements
# pylint: disable=too-many-statements
class CompilerFilter(FilterBase):
    """Filter rules basing on host and device compiler names and versions."""

    def __init__(
        self,
        runtime_infos: Dict[str, Callable[..., bool]] | None = None,
        output: IO[Parameter] | None = None,
    ):
        super().__init__(runtime_infos, output)

    def __call__(
        self,
        row: ParameterValueTuple,
    ) -> bool:
        """Check if given parameter-value-tuple is valid

        Args:
            row (ParameterValueTuple): parameter-value-tuple to verify.

        Returns:
            bool: True, if parameter-value-tuple is valid.
        """
        # uncomment me for debugging
        # print_row_nice(row, bashi_validate=False)

        # Rule: c1
        # NVCC as HOST_COMPILER is not allow
        # this rule will be never used, because of an implementation detail of the covertable
        # library
        # it is not possible to add NVCC as HOST_COMPILER and filter out afterwards
        # this rule is only used by bashi-verify
        if HOST_COMPILER in row and row[HOST_COMPILER].name == NVCC:
            self.reason("nvcc is not allowed as host compiler")
            return False

        if HOST_COMPILER in row and DEVICE_COMPILER in row:
            if NVCC in (row[HOST_COMPILER].name, row[DEVICE_COMPILER].name):
                # Rule: c2
                # related to rule c13
                if row[HOST_COMPILER].name not in (GCC, CLANG):
                    self.reason("only gcc and clang are allowed as nvcc host compiler")
                    return False
            else:
                # Rule: c3
                if row[HOST_COMPILER].name != row[DEVICE_COMPILER].name:
                    self.reason("host and device compiler name must be the same (except for nvcc)")
                    return False

                # Rule: c4
                if row[HOST_COMPILER].version != row[DEVICE_COMPILER].version:
                    self.reason(
                        "host and device compiler version must be the same (except for nvcc)",
                    )
                    return False

        # now idea, how remove nested blocks without hitting the performance
        # pylint: disable=too-many-nested-blocks
        if DEVICE_COMPILER in row and row[DEVICE_COMPILER].name == NVCC:
            if HOST_COMPILER in row and row[HOST_COMPILER].name == GCC:
                # Rule: c5
                # related to rule b10
                # remove all unsupported nvcc gcc version combinations
                # define which is the latest supported gcc compiler for a nvcc version

                # if a nvcc version is not supported by bashi, assume that the version supports the
                # latest gcc compiler version
                if row[DEVICE_COMPILER].version <= NVCC_GCC_MAX_VERSION[0].nvcc:
                    # check the maximum supported gcc version for the given nvcc version
                    for nvcc_gcc_comb in NVCC_GCC_MAX_VERSION:
                        if row[DEVICE_COMPILER].version >= nvcc_gcc_comb.nvcc:
                            if row[HOST_COMPILER].version > nvcc_gcc_comb.host:
                                self.reason(
                                    f"nvcc {row[DEVICE_COMPILER].version} "
                                    f"does not support gcc {row[HOST_COMPILER].version}",
                                )
                                return False
                            break

            if HOST_COMPILER in row and row[HOST_COMPILER].name == CLANG:
                # Rule: c7
                # related to rule b11
                if row[DEVICE_COMPILER].version >= pkv.parse("11.3") and row[
                    DEVICE_COMPILER
                ].version <= pkv.parse("11.5"):
                    self.reason(
                        "clang as host compiler is disabled for nvcc 11.3 to 11.5",
                    )
                    return False

                # Rule: c6
                # related to rule b12
                # remove all unsupported nvcc clang version combinations
                # define which is the latest supported clang compiler for a nvcc version

                # if a nvcc version is not supported by bashi, assume that the version supports the
                # latest clang compiler version
                if row[DEVICE_COMPILER].version <= NVCC_CLANG_MAX_VERSION[0].nvcc:
                    # check the maximum supported gcc version for the given nvcc version
                    for nvcc_clang_comb in NVCC_CLANG_MAX_VERSION:
                        if row[DEVICE_COMPILER].version >= nvcc_clang_comb.nvcc:
                            if row[HOST_COMPILER].version > nvcc_clang_comb.host:
                                self.reason(
                                    f"nvcc {row[DEVICE_COMPILER].version} "
                                    f"does not support clang {row[HOST_COMPILER].version}",
                                )
                                return False
                            break

            # Rule: c15
            # related to rule b9
            if (
                ALPAKA_ACC_GPU_CUDA_ENABLE in row
                and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != row[DEVICE_COMPILER].version
            ):
                self.reason("nvcc and CUDA backend needs to have the same version")
                return False

            # Rule: c16
            # related to rule b14
            if (
                ALPAKA_ACC_GPU_HIP_ENABLE in row
                and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER
            ):
                self.reason("nvcc does not support the HIP backend.")
                return False

            # Rule: c17
            # related to rule b15
            if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version != OFF_VER:
                self.reason("nvcc does not support the SYCL backend.")
                return False

        # Rule: c8
        # related to rule b11
        # clang-cuda 13 and older is not supported
        # this rule will be never used, because of an implementation detail of the covertable
        # library
        # it is not possible to add the clang-cuda versions and filter it out afterwards
        # this rule is only used by bashi-verify
        for compiler in (HOST_COMPILER, DEVICE_COMPILER):
            if (
                compiler in row
                and row[compiler].name == CLANG_CUDA
                and row[compiler].version < pkv.parse("14")
            ):
                self.reason("all clang versions older than 14 are disabled as CUDA Compiler")
                return False

        for compiler in (HOST_COMPILER, DEVICE_COMPILER):
            if compiler in row and row[compiler].name == HIPCC:
                # Rule: c9
                # related to rule b1
                if (
                    ALPAKA_ACC_GPU_HIP_ENABLE in row
                    and row[ALPAKA_ACC_GPU_HIP_ENABLE].version == OFF_VER
                ):
                    self.reason("hipcc requires an enabled HIP backend.")
                    return False

                # Rule: c10
                # related to rule b2
                if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version != OFF_VER:
                    self.reason("hipcc does not support the SYCL backend.")
                    return False

                # Rule: c11
                # related to rule b2
                if (
                    ALPAKA_ACC_GPU_CUDA_ENABLE in row
                    and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER
                ):
                    self.reason("hipcc does not support the CUDA backend.")
                    return False

            if compiler in row and row[compiler].name == ICPX:
                # Rule: c12
                # related to rule b4
                if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version == OFF_VER:
                    self.reason("icpx requires an enabled SYCL backend.")
                    return False

                # Rule: c13
                # related to rule b5
                if (
                    ALPAKA_ACC_GPU_HIP_ENABLE in row
                    and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER
                ):
                    self.reason("icpx does not support the HIP backend.")
                    return False

                # Rule: c14
                # related to rule b6
                if (
                    ALPAKA_ACC_GPU_CUDA_ENABLE in row
                    and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER
                ):
                    self.reason("icpx does not support the CUDA backend.")
                    return False

            if compiler in row and row[compiler].name == CLANG_CUDA:
                # Rule: c15
                # related to rule b16
                if (
                    ALPAKA_ACC_GPU_CUDA_ENABLE in row
                    and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version == OFF_VER
                ):
                    self.reason("clang-cuda requires an enabled CUDA backend.")
                    return False

                if (
                    ALPAKA_ACC_GPU_CUDA_ENABLE in row
                    and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER
                ):
                    # Rule: c16
                    # related to rule b17
                    # if a clang-cuda version is newer than the latest known clang-cuda version,
                    # we needs to assume that it supports every CUDA SDK version
                    # pylint: disable=duplicate-code
                    if row[compiler].version <= CLANG_CUDA_MAX_CUDA_VERSION[0].clang_cuda:
                        # check if know clang-cuda version supports CUDA SDK version
                        for version_combination in CLANG_CUDA_MAX_CUDA_VERSION:
                            if row[compiler].version >= version_combination.clang_cuda:
                                if (
                                    row[ALPAKA_ACC_GPU_CUDA_ENABLE].version
                                    > version_combination.cuda
                                ):
                                    self.reason(
                                        f"clang-cuda {row[compiler].version} does not support "
                                        f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version}.",
                                    )
                                    return False
                                break

                # Rule: c17
                # related to rule b14
                if (
                    ALPAKA_ACC_GPU_HIP_ENABLE in row
                    and row[ALPAKA_ACC_GPU_HIP_ENABLE].version != OFF_VER
                ):
                    self.reason("clang-cuda does not support the HIP backend.")
                    return False

                # Rule: c18
                # related to rule b15
                if ALPAKA_ACC_SYCL_ENABLE in row and row[ALPAKA_ACC_SYCL_ENABLE].version != OFF_VER:
                    self.reason("clang-cuda does not support the SYCL backend.")
                    return False

            if CXX_STANDARD in row:
                for compiler in (HOST_COMPILER, DEVICE_COMPILER):
                    # Rule: c21
                    if _remove_unsupported_compiler_cxx_combination(
                        row, GCC, compiler, GCC_CXX_SUPPORT_VERSION, self.output
                    ):
                        # reason() is inside _remove_unsupported_compiler_cxx_combination
                        return False
                    # Rule: c22
                    if _remove_unsupported_compiler_cxx_combination(
                        row, CLANG, compiler, CLANG_CXX_SUPPORT_VERSION, self.output
                    ):
                        # reason() is inside _remove_unsupported_compiler_cxx_combination
                        return False

                    # Rule: c25
                    if _remove_unsupported_compiler_cxx_combination(
                        row, CLANG_CUDA, compiler, CLANG_CUDA_CXX_SUPPORT_VERSION, self.output
                    ):
                        # reason() is inside _remove_unsupported_compiler_cxx_combination
                        return False

                    # Rule: c28
                    if _remove_unsupported_compiler_cxx_combination(
                        row, ICPX, compiler, ICPX_CXX_SUPPORT_VERSION, self.output
                    ):
                        # reason() is inside _remove_unsupported_compiler_cxx_combination
                        return False

                    # Rule: c29
                    if _remove_unsupported_compiler_cxx_combination(
                        row, HIPCC, compiler, HIPCC_CXX_SUPPORT_VERSION, self.output
                    ):
                        # reason() is inside _remove_unsupported_compiler_cxx_combination
                        return False

                # Rule: c23
                if _remove_unsupported_compiler_cxx_combination(
                    row, NVCC, DEVICE_COMPILER, NVCC_CXX_SUPPORT_VERSION, self.output
                ):
                    # reason() is inside _remove_unsupported_compiler_cxx_combination
                    return False

                if (
                    ALPAKA_ACC_GPU_CUDA_ENABLE in row
                    and row[ALPAKA_ACC_GPU_CUDA_ENABLE].version != OFF_VER
                ):
                    # Rule: c26
                    # With the given CUDA backend version we can restrict the possible C++ standard
                    # already. If there is no Nvcc or Clang-CUDA version which supports the given
                    # C++ standard with the given CUDA SDK, we can return false before the host or
                    # device compiler was added to the row.
                    if row[CXX_STANDARD].version > _get_max_supported_cxx_version_for_cuda_sdk(
                        row[ALPAKA_ACC_GPU_CUDA_ENABLE].version,
                        NVCC_CXX_SUPPORT_VERSION,
                        MAX_CUDA_SDK_CXX_SUPPORT,
                    ):
                        self.reason(
                            f"There is not Nvcc or Clang-CUDA version which supports "
                            f"C++-{row[CXX_STANDARD].version} + CUDA "
                            f"{row[ALPAKA_ACC_GPU_CUDA_ENABLE].version}",
                        )
                        return False

                    # Rule: c27
                    # Normally Clang-CUDA support earlier a new C++ standard for a given CUDA SDK,
                    # than Nvcc. The rule cover the corner case, that Clang-CUDA supports a later
                    # C++ version than Nvcc. But Clang-CUDA does not automatically cover the latest
                    # CUDA SDK version. Therefore there is the case, that a specific CUDA SDK
                    # version supports a higher C++ standard than it successor.
                    # Example: Clang-CUDA 17 supports CUDA 12.1 and C++ 23. Therefore CUDA 12.1 and
                    # C++ 23 is possible. But Clang-CUDA 17 does not support CUDA 12.2. Therefore
                    # CUDA 12.2 can be only compiled with Nvcc and the maximum standard is C++20.
                    if row[
                        CXX_STANDARD
                    ].version > _get_max_supported_cxx_version_for_cuda_sdk_for_nvcc(
                        row[ALPAKA_ACC_GPU_CUDA_ENABLE].version,
                        NVCC_CXX_SUPPORT_VERSION,
                    ) and row[
                        CXX_STANDARD
                    ].version <= _get_max_supported_cxx_version_for_cuda_sdk_for_clang_cuda(
                        row[ALPAKA_ACC_GPU_CUDA_ENABLE].version, MAX_CUDA_SDK_CXX_SUPPORT
                    ):
                        if (
                            row[ALPAKA_ACC_GPU_CUDA_ENABLE].version
                            > CLANG_CUDA_MAX_CUDA_VERSION[0].cuda
                        ):
                            self.reason(
                                f"For the potential combination of C++-{row[CXX_STANDARD].version} "
                                f"+ CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} there is no "
                                f"Clang-CUDA compiler which support this.",
                            )
                            return False
                    # Rule: c24
                    # If we know that the CUDA backend is enabled and the host compiler is GCC or
                    # Clang, the device compiler must be Nvcc.
                    # In this case, we know that the CUDA SDK and Nvcc has the same version number
                    # and therefore we can determine which is the maximum supported C++ standard.
                    if HOST_COMPILER in row and row[HOST_COMPILER].name in (GCC, CLANG):
                        if _remove_unsupported_compiler_cxx_combination(
                            row,
                            ALPAKA_ACC_GPU_CUDA_ENABLE,
                            ALPAKA_ACC_GPU_CUDA_ENABLE,
                            NVCC_CXX_SUPPORT_VERSION,
                            None,
                        ):
                            self.reason(
                                f"{row[HOST_COMPILER].name} {row[HOST_COMPILER].version} + "
                                f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} + "
                                f"C++ {row[CXX_STANDARD].version}: "
                                f"there is no Nvcc version which support this combination",
                            )
                            return False

        return True


@typechecked
def compiler_filter_typechecked(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
) -> bool:
    """Type-checked version of CompilerFilter()(). Type checking has a big performance cost, which
    is why the non type-checked version is used for the pairwise generator.
    """
    return CompilerFilter(output=output)(row)
