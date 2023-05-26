from __future__ import annotations
from typing import Any, Callable, Dict, Optional, List, TypeVar, Union, Generic
from dataclasses import dataclass
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
# MaxFit = sys.float_info.max

FitnessFunction = Callable[[Pos],float]
ConstraintChecker = Callable[[Pos], Pos]
ConstraintsChecker = List[ConstraintChecker]

# @dataclass(frozen=True)
# class FitnessFunctionDef:
#     func: FitnessFunction
#     minimize: bool

@dataclass(frozen=True)
class Bounds:
    min: float
    max: float

@dataclass(frozen=True)
class Evaluation:
    pos: Pos = None
    fit: Fit = np.inf

@dataclass(frozen=True)
class StopCondition:
    fit: Fit
    max_evals: int
    max_gen: int

@dataclass(frozen=True)
class SearchContext:
    evaluate: Callable[[Pos], Evaluation]
    #params: Parameters
    ndims: int
    bounds: Bounds
    curr_gen: int
    max_gen: int
    curr_fit: Fit
    min_fit: Fit
    curr_eval: int
    max_evals: int

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
class AbstractInfo(Generic[L, T]): 
    bounds: Bounds
    ndims: int
    
    def all()->L:
        raise NotImplementedError
    
    def size()->int:
        raise NotImplementedError
    
    def best()->T:
        raise NotImplementedError()
    
    def worse()->T:
        raise NotImplementedError()
    
    def k_best(size: int)->L:
        raise NotImplementedError()

    def k_worse(size: int)->L:
        raise NotImplementedError()
    
    def filter(f: Callable[[T], bool])->L: 
        raise NotImplementedError()

    def pick_random(k: Optional[int] = None, replace: bool = False)->Union[L,T]:
        raise NotImplementedError()
        
    def pick_roulette(k: Optional[int] = None, replace: bool = False)->Union[L,T]:
        raise NotImplementedError()
    
    def min(key: Union[str, Callable[[T], Any]] = "best")->T:
        raise NotImplementedError()

    def max(key: Union[str, Callable[[T], Any]] = "best")->T:
        raise NotImplementedError()

@dataclass(frozen=True)
class GroupInfo(AbstractInfo[AgentList, Agent]): 
    bounds: Bounds
    ndims: int
        
@dataclass(frozen=True)
class IReference: 
    agent: Agent
    
    def is_better(other: IReference)->bool:
        raise NotImplementedError()
    
    def get(key: Union[str, Callable[[IReference], Pos]] = "best")->Pos:
        raise NotImplementedError()

    def add(other: Union[IReference, Pos, int, float], key:Union[str, Callable[[IReference], Pos]] = "best")->Pos: 
        raise NotImplementedError()
    
    def subract(other: Union[IReference, Pos, int, float], key:Union[str, Callable[[IReference], Pos]] = "best")->Pos: 
        raise NotImplementedError()    
    
    def multiply(other: Union[IReference, Pos, int, float], key:Union[str, Callable[[IReference], Pos]] = "best")->Pos: 
        raise NotImplementedError()

    def divide(other: Union[IReference, Pos, int, float], key:Union[str, Callable[[IReference], Pos]] = "best")->Pos: 
        raise NotImplementedError()

    def power(other: Union[IReference, Pos, int, float], key:Union[str, Callable[[IReference], Pos]] = "best")->Pos: 
        raise NotImplementedError()
    
    def modulus(other: Union[IReference, Pos, int, float], key:Union[str, Callable[[IReference], Pos]] = "best")->Pos: 
        raise NotImplementedError()


@dataclass(frozen=True)
class IReferences: 

    def get(index: int)->IReference:
        raise NotImplementedError()
    
    def indices()->List[int]:
        raise NotImplementedError()
    
    def pop()->IReference:
        raise NotImplementedError()
        
    def reduce(accumulator: Callable[[IReference], Pos], initial: Pos = None)->Pos:
        raise NotImplementedError()
    
    def sum(key: Union[str, Callable[[IReference], Pos]] = "best")->Pos:
        raise NotImplementedError()

    def avg(weights: List[float] = None, key: Union[str, Callable[[IReference], Pos]] = "best")->Pos:
        raise NotImplementedError()
    
    def min(key: Union[str, Callable[[IReference], Union[Pos, Fit]]] = "best")->Pos:
        raise NotImplementedError()

    def max(key: Union[str, Callable[[IReference], Union[Pos, Fit]]] = "best")->Pos:
        raise NotImplementedError()
    
    def diff(other: Union[IReference, Pos, int, float],  key: Union[str, Callable[[IReference], Pos]] = "best", reversed: bool = False)->Pos:
       raise NotImplementedError()
    
    def reverse_diff(other: Union[IReference, Pos, int, float],  key: Union[str, Callable[[IReference], Pos]] = "best")->Pos:
       raise NotImplementedError()

    def size()->int:
        raise NotImplementedError()
    

@dataclass(frozen=True)
class PopulationInfo:
    info: GroupInfo
    group_info: List[GroupInfo]

# @dataclass(frozen=True)
# class Population:
#     agents: AgentList
#     topology: Topology #TODO check topology
#     size: int
#     ndims: int
#     bounds: Bounds

PosGenerator = Callable[[SearchContext], Pos]
StaticTopology = List[List[int]]
DynamicTopology =  Callable[[AgentList], StaticTopology]
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
    min: float
    max: float
    value: ParameterValue

class Parameters: 
    def __init__(self):
        self._parameters:Dict[str, Parameter] = {}

    def add(self, name: str, min: float, max: float, value: Union[float, int, ParameterValue]):
        self._parameters[name] = Parameter(
            name, min, max, 
            value if callable(value) else lambda _: value
        )

@dataclass(frozen=True)
class ISwarmContext(AbstractInfo[IReferences, IReference]):
    picked: List[int]
    def pick_random_unique(k: Optional[int] = None, replace: bool = False)->Union[IReference,IReferences]:
        raise NotImplementedError()
        
    def pick_roulette_unique(k: Optional[int] = None, replace: bool = False)->Union[IReference,IReferences]:
        raise NotImplementedError()

@dataclass(frozen=True)
class UpdateContext:
    agent: Agent
    swarm: ISwarmContext
    search_context: SearchContext
    random: Random
    vars: Dict[str, Union[Pos, IReference, IReferences]]

    def param(self, name: str)->float:
        return self.search_context.params[name](self)

    def get(self, name: str)->Union[Pos, IReference, IReferences]:
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

# SelectionMethod = Callable[[GroupInfo], AgentList]
# UpdateCondition = Callable[[UpdateContext], bool]
# UpdateMethod = Callable[[UpdateContext], Agent]
# RecombinationMethod = Callable[[Pos, Pos], Pos]
# Condition = Callable[[Agent], bool]
# OrderingMethod = Callable[[Agent], float]


# @dataclass(frozen=True)
# class SearchSpace:
#     fit_func_def: FitnessFunctionDef
#     ndims: int
#     bounds: Bounds
#     constraints: ConstraintsChecker
   
@dataclass(frozen=True)
class SearchStrategy:
    initialization: Initialization
    parameters: Parameters
    update_pipeline: List[Update]
   
# @dataclass(frozen=True)
# class SearchRequest:
#     strategy: Optional[SearchStrategy]
#     space: Optional[SearchSpace]
#     until: Optional[StopCondition]

SearchResults = List[Evaluation]

# @dataclass(frozen=True)
# class SearchFailed:
#     error: Exception


