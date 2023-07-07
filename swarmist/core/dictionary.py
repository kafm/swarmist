from __future__ import annotations
from typing import Any, Callable, Dict, Optional, List, TypeVar, Union, Generic
from dataclasses import dataclass, astuple
import numpy as np
from swarmist.core.random import Random

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")
L = TypeVar("L")
KeyValue = Dict[K, V]
Pos = List[float]
Fit = float
Number = float | int


@dataclass(frozen=True)
class Bounds:
    min: float | List[float]
    max: float | List[float]


@dataclass(frozen=True)
class Evaluation:
    pos: Pos = None
    fit: Fit = np.inf

    def __iter__(self):
        return iter(astuple(self))


@dataclass(frozen=True)
class StopCondition:
    fit: Fit
    max_evals: int
    max_gen: int


@dataclass(frozen=True)
class SearchContext:
    evaluate: Callable[[Pos], Evaluation]
    parameters: Parameters
    ndims: int
    bounds: Bounds
    curr_gen: int
    max_gen: int
    curr_fit: Fit
    min_fit: Fit
    curr_eval: int
    max_evals: int

    def __getitem__(self, item):
        return getattr(self, item)


FitnessFunction = Callable[[Pos], Fit]
ConstraintChecker = Callable[[Pos], Fit]
ConstraintValue = ConstraintChecker | Fit
ConstraintsChecker = List[ConstraintChecker]


@dataclass(frozen=True)
class Agent:
    index: int
    ndims: int
    delta: Pos
    pos: Pos
    best: Pos
    fit: Fit
    trials: int
    improved: bool

    def __getitem__(self, item):
        return getattr(self, item)


AgentList = List[Agent]


@dataclass(frozen=True)
class AbstractInfo(Generic[L, T]):
    def all(self) -> L:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def best(self) -> T:
        raise NotImplementedError()

    def worse(self) -> T:
        raise NotImplementedError()

    def k_best(self, size: int) -> L:
        raise NotImplementedError()

    def k_worse(self, size: int) -> L:
        raise NotImplementedError()

    def filter(self, f: Callable[[T], bool]) -> L:
        raise NotImplementedError()

    def pick_random(
        self, k: Optional[int] = None, replace: bool = False
    ) -> Union[L, T]:
        raise NotImplementedError()

    def pick_roulette(
        self, k: Optional[int] = None, replace: bool = False
    ) -> Union[L, T]:
        raise NotImplementedError()

    def min(self, key: Union[str, Callable[[T], Any]] = "best") -> T:
        raise NotImplementedError()

    def max(self, key: Union[str, Callable[[T], Any]] = "best") -> T:
        raise NotImplementedError()


@dataclass(frozen=True)
class GroupInfo(AbstractInfo[AgentList, Agent]):
    bounds: Bounds
    ndims: int


@dataclass(frozen=True)
class IReference:
    agent: Agent

    def is_better(self, other: IReference) -> bool:
        raise NotImplementedError()

    def get(self, key: Union[str, Callable[[Agent], Pos]] = "best") -> Pos:
        raise NotImplementedError()

    def add(
        self,
        other: Union[IReference, Pos, int, float],
        key: Union[str, Callable[[IReference], Pos]] = "best",
    ) -> Pos:
        raise NotImplementedError()

    def subtract(
        self,
        other: Union[IReference, Pos, int, float],
        key: Union[str, Callable[[IReference], Pos]] = "best",
    ) -> Pos:
        raise NotImplementedError()

    def multiply(
        self,
        other: Union[IReference, Pos, int, float],
        key: Union[str, Callable[[IReference], Pos]] = "best",
    ) -> Pos:
        raise NotImplementedError()

    def divide(
        self,
        other: Union[IReference, Pos, int, float],
        key: Union[str, Callable[[IReference], Pos]] = "best",
    ) -> Pos:
        raise NotImplementedError()

    def power(
        self,
        other: Union[IReference, Pos, int, float],
        key: Union[str, Callable[[IReference], Pos]] = "best",
    ) -> Pos:
        raise NotImplementedError()

    def modulus(
        self,
        other: Union[IReference, Pos, int, float],
        key: Union[str, Callable[[IReference], Pos]] = "best",
    ) -> Pos:
        raise NotImplementedError()

    def __add__(self, other: Union[IReference, Pos, int, float]) -> Pos:
        return self.add(other)

    def __sub__(self, other: Union[IReference, Pos, int, float]) -> Pos:
        return self.subtract(other)

    def __mul__(self, other: Union[IReference, Pos, int, float]) -> Pos:
        return self.multiply(other)

    def __div__(self, other: Union[IReference, Pos, int, float]) -> Pos:
        raise self.divide(other)

    def __pow__(self, other: Union[IReference, Pos, int, float]) -> Pos:
        return self.power(other)

    def __mod__(self, other: Union[IReference, Pos, int, float]) -> Pos:
        return self.modulus(other)


@dataclass(frozen=True)
class IReferences:
    def get(self, index: int) -> IReference:
        raise NotImplementedError()
    
    def all(self) -> List[IReference]:
        raise NotImplementedError()

    def indices(self) -> List[int]:
        raise NotImplementedError()

    def pop(self) -> IReference:
        raise NotImplementedError()

    def reduce(
        self, accumulator: Callable[[IReference], Pos], initial: Pos = None
    ) -> Pos:
        raise NotImplementedError()

    def sum(self, key: Union[str, Callable[[IReference], Pos]] = "best") -> Pos:
        raise NotImplementedError()

    def avg(
        self,
        weights: List[float] = None,
        key: Union[str, Callable[[IReference], Pos]] = "best",
    ) -> Pos:
        raise NotImplementedError()

    def min(
        self, key: Union[str, Callable[[IReference], Union[Pos, Fit]]] = "best"
    ) -> Pos:
        raise NotImplementedError()

    def max(
        self, key: Union[str, Callable[[IReference], Union[Pos, Fit]]] = "best"
    ) -> Pos:
        raise NotImplementedError()

    def diff(
        self,
        other: Union[IReference, Pos, int, float],
        key: Union[str, Callable[[IReference], Pos]] = "best",
        reversed: bool = False,
    ) -> Pos:
        raise NotImplementedError()

    def reverse_diff(
        self,
        other: Union[IReference, Pos, int, float],
        key: Union[str, Callable[[IReference], Pos]] = "best",
    ) -> Pos:
        raise NotImplementedError()

    def size(self) -> int:
        raise NotImplementedError()


@dataclass(frozen=True)
class PopulationInfo:
    info: GroupInfo
    group_info: List[GroupInfo]


PosGenerator = Callable[[SearchContext], Pos]
StaticTopology = List[List[int]]
DynamicTopology = Callable[[AgentList], StaticTopology]
Topology = StaticTopology | DynamicTopology
TopologyBuilder = Callable[[AgentList], Topology]


@dataclass(frozen=True)
class Initialization:
    population_size: int
    generate_pos: PosGenerationMethod
    topology: TopologyBuilder


PosGenerationMethod = Callable[[SearchContext], Pos]
ParameterValue = Callable[[SearchContext], float]


@dataclass(frozen=True)
class Parameter:
    name: str
    value: ParameterValue
    min: Optional[float]
    max: Optional[float]


class Parameters:
    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}

    def add(
        self,
        name: str,
        value: Union[float, int, ParameterValue],
        min: Optional[float] = None,
        max: Optional[float] = None,
    ):
        self._parameters[name] = Parameter(
            name, value if callable(value) else lambda _: value, min, max
        )

    def get(self, name: str, ctx: SearchContext) -> float:
        param = self._parameters[name]
        return np.clip(param.value(ctx), param.min, param.max)
    
    def param(self, name: str)->Parameter:
        return self._parameters[name]

    def __repr__(self):
        return f"Parameters({self._parameters})"


@dataclass(frozen=True)
class ISwarmContext(AbstractInfo[IReferences, Union[IReference, Agent]]):
    picked: List[int]

    def pick_random_unique(
        self, k: Optional[int] = None, replace: bool = False
    ) -> Union[IReference, IReferences]:
        raise NotImplementedError()

    def pick_roulette_unique(
        self, k: Optional[int] = None, replace: bool = False
    ) -> Union[IReference, IReferences]:
        raise NotImplementedError()


@dataclass(frozen=True)
class UpdateContext:
    agent: Agent
    swarm: ISwarmContext
    search_context: SearchContext
    random: Random
    vars: Dict[str, Union[Pos, IReference, IReferences]]

    def param(self, name: str) -> float:
        return self.search_context.parameters.get(name, self)

    def get(self, name: str) -> Union[Pos, IReference, IReferences]:
        if name not in self.vars:
            return self.search_context[name.lower()]
        return self.vars[name]


Selection = Callable[[GroupInfo], AgentList]
Condition = Callable[[Agent], bool]
Order = Callable[[Agent], Any]
PosEditor = Callable[[UpdateContext], Pos]
Recombination = Callable[[Agent, Pos], Agent]


@dataclass(frozen=True)
class Update:
    selection: Selection
    condition: Condition
    recombination: Recombination
    editor: PosEditor


@dataclass(frozen=True)
class SearchStrategy:
    initialization: Initialization
    parameters: Parameters
    update_pipeline: List[Update]


SearchResults = List[Evaluation]
