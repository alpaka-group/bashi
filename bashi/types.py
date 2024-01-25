"""bashi data types"""

from typing import TypeAlias, Tuple, List, Callable
from collections import OrderedDict
from packaging.version import Version

Parameter: TypeAlias = str
ValueName: TypeAlias = str
ValueVersion: TypeAlias = Version
ParameterValue: TypeAlias = Tuple[ValueName, ValueVersion]
ParameterValueList: TypeAlias = List[ParameterValue]
ParameterValueMatrix: TypeAlias = OrderedDict[Parameter, ParameterValueList]
ParameterValuePair: TypeAlias = OrderedDict[Parameter, ParameterValue]
ParameterValueTuple: TypeAlias = OrderedDict[Parameter, ParameterValue]
Combination: TypeAlias = OrderedDict[Parameter, ParameterValue]
CombinationList: TypeAlias = List[Combination]

# function signature of a filter function
FilterFunction: TypeAlias = Callable[[OrderedDict[str, Tuple[str, Version]]], bool]
