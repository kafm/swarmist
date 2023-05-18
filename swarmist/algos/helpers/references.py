from __future__ import annotations
from typing import Callable, List
from dataclasses import dataclass
from swarmist.core.dictionary import UpdateContext, Agent, AgentList, Pos, Fit
from swarmist.utils import random
import numpy as np

def best_neighbor()->Callable[[UpdateContext], Reference]:
    return k_best_neighbors(1)

def k_best_neighbors(size: int= 1)->Callable[[UpdateContext], Reference]:
    return lambda ctx: Reference.of(ctx.best(size))

def worse_neighbor()->Callable[[UpdateContext], Reference]:
    return k_worse_neighbors(1)

def k_worse_neighbors(size: int= 1)->Callable[[UpdateContext], Reference]:
    return lambda ctx: Reference.of(ctx.worse(size))

def all_neighbors()->Callable[[UpdateContext], Reference]:
    return lambda ctx: Reference.of(ctx.all())

def rand_to_best(f: float = .5)->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        best: Agent = ctx.best(1)[0]
        a: Agent =  ctx.pick_random_unique(1)[0]
        pos = f * best.best + (1 - f) * a.best
        return Reference.of([best, a], default=pos)
    return callback

def current_to_best(f: float = .5)->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        best: Agent = ctx.best(1)[0]
        current: Agent =  ctx.agent
        pos = current + f * (best.best - current.best)
        return Reference.of([best, current], default=pos)
    return callback

def pick_random(size: int = 1, replace:bool = False)->Callable[[UpdateContext], Reference]:
    return lambda ctx: Reference.of(ctx.pick_random(size, replace=replace))

def pick_random_unique(size: int = 1, replace:bool = False)->Callable[[UpdateContext], Reference]:
    return lambda ctx: Reference.of(ctx.pick_random_unique(size, replace=replace))

def pick_roulette(size: int = 1, replace:bool = False)->Callable[[UpdateContext], Reference]:
    return lambda ctx: Reference.of(ctx.pick_roulette(size, replace=replace))
    
def pick_roulette_unique(size: int = 1, replace:bool = False)->Callable[[UpdateContext], Reference]:
    return lambda ctx: Reference.of(ctx.pick_roulette_unique(size, replace=replace))

def self_best()->Callable[[UpdateContext], Reference]:
    return lambda ctx: Reference.of([ctx.agent], default=ctx.agent.best)

def self_pos()->Callable[[UpdateContext], Reference]:
    return lambda ctx: Reference.of([ctx.agent], default=ctx.agent.pos)

def random_pos()->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        pos = random.uniform(low=ctx.bounds.min, high=ctx.bounds.max, size= ctx.agent.ndims)
        return Reference.of([ctx.agent], default=pos)
    return callback

DefaultRefPos = str | Pos


#TODO check better way to deal with  reference composed of one agent and reference composed by more than one agent
#TODO make simpler and better
@dataclass(frozen=True)
class Reference: 
    agents: AgentList
    pos_list: List[Pos]
    default_arr: DefaultRefPos

    @classmethod
    def of(cls, agents: AgentList, default: DefaultRefPos = "avg")->Reference:
        pos_list = [a.best for a in agents]
        return cls(agents, pos_list, default)
    
    def avg(self, transformer: Callable[[Pos], Pos] = None)->Pos:
        pos_list = self.pos_list if not transformer else list(map(transformer, self.pos_list))
        return np.average(pos_list, axis=0)

    def sum(self)->Pos:
        return np.sum(self.pos_list, axis=0)
    
    def best(self)->Reference: 
        return Reference.of([min(self.agents, key=lambda a: a.fit)])
    
    def fit(self)->Reference:
        return self.best_agent().fit
    
    def first(self)->Reference:
        return Reference.of(self.agents[:1])
    
    def pos(self) -> Pos: 
        if isinstance(self.default_arr, (list, np.ndarray)):
            return np.array(self.default_arr)
        return self.sum() if self.default_arr == "sum" else self.avg() 
    
    def best_agent(self)->Agent:
        return min(self.agents, key=lambda a: a.fit)
    
    def __array__(self) -> np.ndarray:
        return self.pos()
