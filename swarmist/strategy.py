from __future__ import annotations
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from pymonad.either import Either, Right, Left
from swarmist.core.dictionary import PosGenerationMethod, TopologyBuilder, ParameterValue, Parameters, SearchStrategy, SearchContext, Initialization
from swarmist.initialization import InitializationMethods, TopologyMethods
from swarmist.update import UpdateBuilder

class Strategy:
    def __init__(self):
        self._parameters = Parameters()
        self._initialization = InitializationMethods().random()
        self._topology = TopologyMethods().gbest()
        self._population_size = 20
        self._pipeline_builders: List[UpdateBuilder] = []

    def param(self, name: str, min: float, max: float, value: Optional[ParameterValue]=None)->Strategy:
        self._parameters.add(name, min, max, value)
        return self
    
    def get_param(self, name: str, ctx: SearchContext = None)->Optional[ParameterValue]:   
        return self._parameters.get(name, ctx)

    def init(self, initialization: PosGenerationMethod, size: int)->Strategy:
        self._initialization = initialization
        self._population_size = size
        return self

    def topology(self, topology: TopologyBuilder)->Strategy:
        self._topology = topology
        return self

    def pipeline(self, *updates: UpdateBuilder)->Strategy:
        self._pipeline_builders = list(updates)
        return self
    
    def population_size(self)->int:
        return self._population_size


    def get(self)->Either[Exception, SearchStrategy]:
        try:
            return Right(
                SearchStrategy(
                    initialization=Initialization(
                        population_size=self._population_size,
                        generate_pos=self._initialization,
                        topology=self._topology
                    ),
                    parameters=self._parameters,
                    update_pipeline=[
                        builder.get() for builder in self._pipeline_builders if builder
                    ]
                )
            )
        except Exception as e:
            return Left(e)
    