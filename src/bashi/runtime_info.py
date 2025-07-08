"""Filter rules which will generated during runtime depending on the input of the input
parameter-value-matrix"""

from typing import List, Dict
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from bashi.versions import UbuntuHipMinMax
from bashi.types import ValueVersion


def get_hip_sdk_supporting_ubuntus(
    ubuntus: List[ValueVersion],
    hipccs: List[ValueVersion],
    ubuntu_hip_version_range: List[UbuntuHipMinMax],
):
    """Take a list of given Ubuntu and HIP SDK versions and also a list of which HIP SDK can be
    installed on which Ubuntu. Creates a validator object, which checks if a given Ubuntu version is
    in a list. The list contains a Ubuntu version if:

    - The Ubuntu version is in the argument ubuntus and
    - a version range is defined for the Ubuntu version in the argument ubuntu_hip_version_range and
    - there is a least one HIP SDK version in the arguments hipccs which matches the hip version
        range of a given ubuntu hip version range pair.

    Args:
        ubuntus (List[ValueVersion]): List of Ubuntu versions.
        hipccs (List[ValueVersion]): List of HIP SDK versions
        ubuntu_hip_version_range (List[UbuntuHipMinMax]): List of supported HIP SDK versions for
            given Ubuntu versions.

    Raises:
        RuntimeError: If at least one of the input lists is empty

    Returns:
        ValidUbuntuHip: An validator object with a call operator, which takes a Ubuntu version and
            returns True if the version is in the filtered list.
    """
    if not ubuntus or not hipccs or not ubuntu_hip_version_range:
        raise RuntimeError(
            "It is not supported if arguments ubuntus, hipccs and ubuntu_hip_version_range "
            "are empty."
        )

    # pylint: disable=too-few-public-methods
    class ValidUbuntuHip:
        """Check if a given Ubuntu version is in the valid_ubuntu list."""

        def __init__(self, valid_ubuntus: List[Version]):
            self.valid_ubuntus = valid_ubuntus

        def __call__(self, ubuntu_ver: Version) -> bool:
            return ubuntu_ver in self.valid_ubuntus

    # store which ubuntu version is valid
    valid_ubuntus: Dict[ValueVersion, bool] = {}
    for ub in ubuntus:
        valid_ubuntus[ub] = True

    # parse in a better suited data structure
    supported_ubuntus: Dict[ValueVersion, SpecifierSet] = {}
    for ubuntu, hip_range in ubuntu_hip_version_range:
        supported_ubuntus[ubuntu] = hip_range

    # disable all ubuntu versions, which are not named in the ubuntu HIP SDK support list
    for ub_ver in valid_ubuntus:
        if ub_ver not in supported_ubuntus:
            valid_ubuntus[ub_ver] = False

    # disable all ubuntu versions, where no HIP SDK version is available for installation
    for ub_ver, valid in valid_ubuntus.items():
        if valid:
            found = False
            for hipcc_ver in hipccs:
                if hipcc_ver in supported_ubuntus[ub_ver]:
                    found = True
                    break

            if not found:
                valid_ubuntus[ub_ver] = False

    return ValidUbuntuHip([ub for ub, valid in valid_ubuntus.items() if valid])
