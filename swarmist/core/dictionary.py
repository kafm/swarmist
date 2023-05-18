from __future__ import annotations
from typing import Callable, Dict, Optional, List, TypeVar
from dataclasses import dataclass
import sys

K = TypeVar("K")
V = TypeVar("V")

KeyValue = Dict[K, V]

Pos = List[float]
Fit = float
MaxFit = sys.float_info.max

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
    fit: Fit = MaxFit

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

@dataclass(frozen=True)
class GroupInfo: 
    bounds: Bounds
    ndims: int
    #fits: List[Fit]
    #probs: List[float]
    
    def all()->AgentList:
        raise NotImplementedError
    
    def size()->int:
        raise NotImplementedError
    
    def best(size: int = 1)->AgentList:
        raise NotImplementedError()
    
    def worse(size: int = 1)->AgentList:
        raise NotImplementedError()
    
    def filter(f: Callable[[Agent], bool])->AgentList: 
        raise NotImplementedError()

    def pick_random(k: int = 1, replace: bool = False)->AgentList:
        raise NotImplementedError()
        
    def pick_roulette(k: int = 1, replace: bool = False)->AgentList:
        raise NotImplementedError()

@dataclass(frozen=True)
class PopulationInfo:
    info: GroupInfo
    group_info: List[GroupInfo]

@dataclass(frozen=True)
class Population:
    agents: AgentList
    topology: Topology #TODO check topology
    size: int
    ndims: int
    bounds: Bounds

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

    def pick_random_unique(self, k: Optional[int] = None, replace: Optional[bool] = False)->AgentList:
        raise NotImplementedError
    
    def pick_roulette_unique(self, k: Optional[int] = None, replace: Optional[bool] = True)->AgentList:
        raise NotImplementedError

SelectionMethod = Callable[[GroupInfo], AgentList]
UpdateCondition = Callable[[UpdateContext], bool]
UpdateMethod = Callable[[UpdateContext], Agent]
RecombinationMethod = Callable[[Pos, Pos], Pos]
Condition = Callable[[Agent], bool]
OrderingMethod = Callable[[Agent], float]



@dataclass(frozen=True)
class Update:
    method: UpdateMethod
    selection: SelectionMethod
    where: Condition = None


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
