from __future__ import annotations
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from swarmist_bck.core.dictionary import PosGenerationMethod, TopologyBuilder

class Metaheuristic:
    def __init__(self):
        self.parameters = Parameters()
        self.defaults = Defaults()

    def param(self, name: str, min: float, max: float, default: Optional[float]=None):
        self.parameters.add(name, min, max, default)

    def pipeline(self, *steps: Step):
        pass

    def __call__(self, **kwargs):
        pass


class Parameters: 
    def __init__(self):
        self._parameters:Dict[str, Parameter] = {}

    def add(self, name: str, min: float, max: float, default: float):
        self._parameters[name] = Parameter(name, min, max, default)

@dataclass(frozen=True)
class Parameter:
    name: str
    min: float
    max: float
    default: float | Callable[..., float]

@dataclass
class Defaults:
    initialization: PosGenerationMethod = None
    topology: TopologyBuilder = None
    