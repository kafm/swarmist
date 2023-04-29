from __future__ import annotations
from typing import Callable, Dict, Optional, List, Union, TypeVar
from dataclasses import dataclass, field
from oslash import Either, Maybe, Nothing

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

KeyValue = Dict[K, V]

Pos = List[float]
#Size = Either[int, Exception]

### SEARCH SPACE ###

FitnessFunction = Callable[[Pos],float]

@dataclass(frozen=True)
class FitnessFunctionDef:
    func: FitnessFunction
    minimize: bool = True

@dataclass(frozen=True)
class Bounds:
    min: float
    max: float

@dataclass(frozen=True) 
class Evaluation:
    pos: Pos
    fit: float

@dataclass(frozen=True) 
class Evaluation:
    pos: Pos
    fit: float

ConstraintsChecker = Callable[[Pos], Pos]

@dataclass(frozen=True)
class SearchSpace:
    fit_func: FitnessFunctionDef
    ndims: int
    bounds: Bounds
    constraints: ConstraintsChecker

#SEARCH

Evaluator = Callable[[Pos],Either[Evaluation, Exception]]

@dataclass(frozen=True)
class StopCondition:
    fit: float
    max_evals: int
    max_gen: int

@dataclass(frozen=True)
class SearchContext:
    evaluate: FitnessFunction
    clip: Pos
    ndims: int
    bounds: Bounds

@dataclass(frozen=True)
class SearchRequest:
    strategy: SearchStrategy
    space: SearchSpace
    until: StopCondition

@dataclass(frozen=True)
class SearchResults:
    results: List[Evaluation]
    error: Exception

### SEARCH STRATEGY ###

Fit = float
Fits = List[Fit]
Prob = float
Probs = List[Prob]

@dataclass(frozen=True)
class Agent: 
    delta: Pos
    pos: Pos
    best: Pos
    fit: Fit
    trials: int
    improved: bool

A = TypeVar('A', bound=Agent)

AgentList = List[A]
AgentMatrix = List[AgentList]

@dataclass(frozen=True)
class GroupInfo: 
    all: Callable[[], AgentList] 
    size: Callable[[], int]
    fits: Fits
    probs: Probs
    best: Callable[[Optional[int]], A]
    worse: Callable[[Optional[int]], A]
    filter: Callable[[Callable[[A,Optional[int]], bool]], AgentList]
    pick_random: Callable[[Optional[int]], Union[A, AgentList]]
    pick_roulette: Callable[[Optional[int]], Union[A, AgentList]]
    map: Callable[[A], T]
    reduce: Callable[[A], T]

@dataclass(frozen=True)
class AgentGroup:
    agents: AgentList
    rank: Callable([], GroupInfo)

@dataclass(frozen=True)
class Population:
    agents: AgentList
    groups: List[AgentGroup]
    size: int
    rank: Callable([], GroupInfo)


InitializationMethod = Callable[[int, SearchContext], AgentList]
StaticTopology = AgentMatrix
DynamicTopology =  Callable[[Optional[AgentList], Optional[StaticTopology]], StaticTopology]
Topology = StaticTopology | DynamicTopology
TopologyBuilder = Callable[[AgentList], Topology]

@dataclass(frozen=True)
class Init:
    population_size: int
    method: InitializationMethod
    topology: Topology

SelectionMethod = Callable[[GroupInfo], AgentList]
UpdateCondition = Maybe[Callable[[Agent], bool]]

@dataclass(frozen=True)
class UpdateContext:
    agent: Agent
    info: GroupInfo
    parameters: Dict
    condition: UpdateCondition = Nothing()
    apply: Callable([Agent], Agent)

UpdateMethod = Callable[[UpdateContext], Agent]
Limit = int
Condition = Callable[[Agent], bool]
Conditions = List[Condition]

@dataclass(frozen=True)
class Update:
    method: UpdateMethod
    selection: SelectionMethod
    where: Conditions = None
    limit: Limit = None

@dataclass(frozen=True)
class SearchStrategy:
    initialization: Init
    update_pipeline: List[Update]

