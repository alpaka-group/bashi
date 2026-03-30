"""Contains relationships with ubuntu and other parameter-values."""

from typing import List, NamedTuple
import packaging.version
import packaging.specifiers
from bashi.version.dependencies.base_version_support import VersionSupportBase


# pylint: disable=too-few-public-methods
class SDKUbuntuSupport(VersionSupportBase):
    """Contains a SDK version and Ubuntu version. Does automatically parse the input strings
    to package.version.Version.

    Provides comparision operators for sorting.
    """

    def __init__(self, sdk_version: str, ubuntu_version: str):
        VersionSupportBase.__init__(self, sdk_version, ubuntu_version)
        self.sdk: packaging.version.Version = self.version1
        self.ubuntu: packaging.version.Version = self.version2


UbuntuSDKMinMax = NamedTuple(
    "UbuntuSDKMinMax",
    [("ubuntu", packaging.version.Version), ("sdk_range", packaging.specifiers.SpecifierSet)],
)


# the list minimum HIP SDK version which can be installed on a specific Ubuntu version
# the next entry in the list defines exclusive, upper bound of a HIP SDK version range
HIP_MIN_UBUNTU: List[SDKUbuntuSupport] = [
    SDKUbuntuSupport("5.0", "20.04"),
    SDKUbuntuSupport("6.0", "22.04"),
    SDKUbuntuSupport("6.3", "24.04"),
]

# the list minimum CUDA SDK version which can be installed on a specific Ubuntu version
# the next entry in the list defines exclusive, upper bound of a HIP SDK version range
CUDA_MIN_UBUNTU: List[SDKUbuntuSupport] = [
    SDKUbuntuSupport("10.0", "18.04"),
    SDKUbuntuSupport("11.0", "20.04"),
    SDKUbuntuSupport("12.0", "24.04"),
]
