"""bashi data types"""

from typing import TypeAlias, List, NamedTuple, Tuple, Union
from collections import OrderedDict
from packaging.version import Version

Parameter: TypeAlias = str
ValueName: TypeAlias = str
ValueVersion: TypeAlias = Version
ParameterValue = NamedTuple("ParameterValue", [("name", ValueName), ("version", ValueVersion)])
ParameterValueList: TypeAlias = List[ParameterValue]
ParameterValueMatrix: TypeAlias = OrderedDict[Parameter, ParameterValueList]
ParameterValueSingle = NamedTuple(
    "ParameterValueSingle", [("parameter", Parameter), ("parameterValue", ParameterValue)]
)
ParameterValuePair = NamedTuple(
    "ParameterValuePair",
    [("first", ParameterValueSingle), ("second", ParameterValueSingle)],
)
ParameterValueTuple: TypeAlias = OrderedDict[Parameter, ParameterValue]
Combination: TypeAlias = OrderedDict[Parameter, ParameterValue]
CombinationList: TypeAlias = List[Combination]

ParsableValueVersion: TypeAlias = Union[str, int, float, Version]

RegularParsableParameterSingle: TypeAlias = Tuple[str, ParsableValueVersion]
CompilerParsableParameterSingle: TypeAlias = Tuple[str, str, ParsableValueVersion]
ParsableParameterSingle: TypeAlias = Union[
    RegularParsableParameterSingle, CompilerParsableParameterSingle
]
