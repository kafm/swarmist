from __future__ import annotations
from typing import Callable, Dict, Optional, List, TypeVar, Tuple, Union
from dataclasses import dataclass, astuple
import sys

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")

KeyValue = Dict[K, V]

Pos = List[float]
Fit = float

FitnessFunction = Callable[[Pos],float]
ConstraintsChecker = Callable[[Pos], Pos]

@dataclass(frozen=True)
class FitnessFunctionDef:
    func: FitnessFunction
    minimize: bool

@dataclass(frozen=True)
class Bounds:
    min: float
    max: float

@dataclass(frozen=True)
class Evaluation:
    pos: Pos = None
    fit: Fit = sys.float_info.max

@dataclass(frozen=True)
class StopCondition:
    fit: Fit
    max_evals: int
    max_gen: int
    
@dataclass(frozen=True)
class SearchContext:
    evaluate: FitnessFunction
    clip: Pos
    ndims: int
    bounds: Bounds


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

AgentList = List[Agent]
OneOrMoreAgents = Agent | AgentList
T = TypeVar("T")

@dataclass(frozen=True)
class GroupInfo: 
    all: Callable[[], AgentList] 
    size: Callable[[], int]
    fits: List[Fit]
    probs: List[float]
    best: Callable[[Optional[int]], AgentList]
    worse: Callable[[Optional[int]], AgentList]
    filter: Callable[[Callable[[Agent,Optional[int]], bool]], AgentList]
    pick_random: Callable[[Optional[int],Optional[bool]], OneOrMoreAgents]
    pick_roulette: Callable[[Optional[int],Optional[bool]], OneOrMoreAgents]
    map: Callable[[Agent], T]

@dataclass(frozen=True)
class AgentGroup:
    agents: AgentList
    rank: Callable(..., GroupInfo)

@dataclass(frozen=True)
class PopulationInfo:
    info: GroupInfo
    group_info: List[GroupInfo]

@dataclass(frozen=True)
class Population:
    agents: AgentList
    topology: Topology #TODO check topology
    size: int

PosGenerationMethod = Callable[[SearchContext], Pos]
StaticTopology = List[List[int]]
DynamicTopology =  Callable[[AgentList], StaticTopology]
Topology = StaticTopology | DynamicTopology
TopologyBuilder = Callable[[AgentList], Topology]

@dataclass(frozen=True)
class Initialization:
    population_size: int
    generate_pos: PosGenerationMethod
    topology: TopologyBuilder

@dataclass(frozen=True)
class UpdateContext(GroupInfo):
    agent: Agent

SelectionMethod = Callable[[GroupInfo], AgentList]
UpdateCondition = Callable[[UpdateContext], bool]
UpdateMethod = Callable[[UpdateContext], Agent]
RecombinationMethod = Callable[[Pos, Pos], Pos]
Condition = Callable[[Agent], bool]

@dataclass(frozen=True)
class Update:
    method: UpdateMethod
    selection: SelectionMethod
    where: Condition


@dataclass(frozen=True)
class SearchSpace:
    fit_func_def: FitnessFunctionDef
    ndims: int
    bounds: Bounds
    constraints: ConstraintsChecker
   
@dataclass(frozen=True)
class SearchStrategy:
    initialization: Initialization
    update_pipeline: List[Update]
   
@dataclass(frozen=True)
class SearchRequest:
    strategy: Optional[SearchStrategy]
    space: Optional[SearchSpace]
    until: Optional[StopCondition]

SearchResults = List[Evaluation]

@dataclass(frozen=True)
class SearchFailed:
    error: Exception
