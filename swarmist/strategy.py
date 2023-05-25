from __future__ import annotations
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from swarmist.core.dictionary import PosGenerationMethod, TopologyBuilder, ParameterValue, Parameters

class Strategy:
    def __init__(self):
        self._parameters = Parameters()
        self._defaults = Defaults()

    def param(self, name: str, min: float, max: float, value: Optional[ParameterValue]=None):
        self._parameters.add(name, min, max, value)

    def init(self, initialization: PosGenerationMethod):
        self._defaults.initialization = initialization

    def topology(self, topology: TopologyBuilder):
        self._defaults.topology = topology

    def pipeline(self, *updates: Step): #TODO
        pass

    def __call__(self, **kwargs):
        pass


@dataclass
class Defaults:
    initialization: PosGenerationMethod = None
    topology: TopologyBuilder = None
    