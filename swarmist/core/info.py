from __future__ import annotations
from typing import Optional, List, Callable, Union, Any
from dataclasses import dataclass
import numpy as np
from swarmist.core.dictionary import Agent, AgentList, Bounds, Fit, GroupInfo, UpdateContext, SearchContext, ISwarmContext, Pos, IReference, IReferences
from swarmist.core.random import Random
from swarmist.core.references import Reference, References

def get_fits(agents: AgentList)->List[Fit]:
    return [a.fit for a in agents]

def fit_to_prob(fits: List[Fit])->List[float]:
    min_fit = min(fits)
    max_fit = max(fits)
    total = 0
    n = len(fits)
    nfits = np.zeros(n)
    if min_fit == max_fit: 
        return np.full(n, 1/n)
        #TODO investigate: print(f"fits={fits}")
    for i in range(n):
        fit = (max_fit - fits[i]) / (max_fit - min_fit)
        nfits[i] = fit
        total += fit
    return np.divide(nfits, total)

@dataclass(frozen=True)
class AgentsInfo(GroupInfo):
    agents: AgentList
    fits: List[Fit]
    probs: List[float]
    rank: List[int]
    gsize: int

    @classmethod
    def of(cls, agents: AgentList, bounds: Bounds, ndims: int)->AgentsInfo:
        fits = get_fits(agents)
        probs = fit_to_prob(fits)
        gsize = len(agents)
        rank = sorted(np.arange(gsize), key=lambda i: fits[i])
        return cls(
            agents=agents, fits=fits, probs=probs, rank=rank,
            gsize=gsize, ndims=ndims, bounds=bounds
        )

    def all(self)->AgentList:
        return self.agents

    def size(self)->int: 
        return self.gsize
    
    def filter(self, f: Callable[[Agent], bool])->AgentList: 
        return list(filter(f,self.agents))
    
    def best(self)->Agent:
        return self.agents[self.rank[0]] 
    
    def worse(self)->Agent:
        return self.agents[self.rank[-1]] 
    
    def k_best(self, k: int)->AgentList:
        return [self.agents[i] for i in self.rank[:k]]

    def k_worse(self, k: int)->AgentList:
        return [self.agents[i] for i in self.rank[-k:]]

    def pick_random(self, k: int = None, replace: bool = False, exclude: List[int] = None)->Union[Agent,AgentList]:
        return (
            np.random.choice(self.agents, size=k, replace=replace) 
            if not exclude 
            else np.random.choice([a for a in self.agents if a.index not in exclude], size=k, replace=replace)
        )
        
    def pick_roulette(self, k: int = None, replace: bool = False, exclude: List[int] = None)->Union[Agent,AgentList]:
        return (
            np.random.choice(self.agents, size=k, replace=replace, p=self.probs)
            if not exclude 
            else np.random.choice([a for a in self.agents if a.index not in exclude], size=k, replace=replace, p=self.probs)
        )
    
    def min(self, key: Union[str, Callable[[Agent], Any]] = "fit")->Agent:
        return min(self.agents, key=key if callable(key) else lambda a: getattr(a, key))

    def max(self,key: Union[str, Callable[[Agent], Any]] = "fit")->Agent:
        return max(self.agents, key=key if callable(key) else lambda a: getattr(a, key))

@dataclass(frozen=True)
class SwarmContext(ISwarmContext):
    info : GroupInfo

    @classmethod
    def of(cls, info: AgentsInfo, picked: List[int] = [])->SwarmContext:
        return cls(info=info, picked=picked)

    def all(self)->References:
        return References.of(self.info.all())
    
    def size(self)->int:
        return self.info.size()
    
    def best(self)->Reference:
        return Reference(self.info.best())
    
    def worse(self)->Reference:
        return Reference(self.info.worse())
    
    def k_best(self, size: int)->References:
        return References.of(self.info.k_best(size))

    def k_worse(self, size: int)->References:
        return References.of(self.info.k_worse(size))
    
    def filter(self, f: Callable[[Agent], bool])->References: 
        return References.of(self.info.filter(f))
    
    def min(self, key: Union[str, Callable[[Agent], Any]] = "fit")->Reference:
        return Reference.of(self.info.min(key))

    def max(self,key: Union[str, Callable[[Agent], Any]] = "fit")->Reference:
        return Reference.of(self.info.max(key))

    def pick_random(self,k: Optional[int] = None, replace: bool = False)->Union[Reference, References]:
        ref = (
            Reference(self.info.pick_random(replace=replace))
            if not k else
            References.of(self.info.pick_random(k=k, replace=replace))
        )
        self._append_picked(ref)
        return ref
        
    def pick_roulette(self, k: Optional[int] = None, replace: bool = False)->Union[Reference, References]:
        ref = (
            Reference(self.info.pick_roulette(replace=replace))
            if not k else
            References.of(self.info.pick_roulette(k=k, replace=replace))
        )
        self._append_picked(ref)
        return ref
    
    def pick_random_unique(self,k: Optional[int] = None, replace: bool = False)->Union[Reference, References]:
        ref = (
            Reference(self.info.pick_random(replace=replace, exclude=self.picked))
            if not k else
            References.of(self.info.pick_random(k=k, replace=replace, exclude=self.picked))
        )
        self._append_picked(ref)
        return ref
        
    def pick_roulette_unique(self, k: Optional[int] = None, replace: bool = False)->Union[Reference, References]:
        ref = (
            Reference(self.info.pick_roulette(replace=replace, exclude=self.picked))
            if not k else
            References.of(self.info.pick_roulette(k=k, replace=replace, exclude=self.picked))
        )
        self._append_picked(ref)
        return ref
    
    def _append_picked(self, ref: Union[Reference, References]):
        if isinstance(ref, Reference):
            self.picked.append(ref.agent.index)
        else: 
            self.picked.extend([r.index for r in ref])
    
@dataclass(frozen=True)
class UpdateInfo(UpdateContext):
  
    @classmethod
    def of(cls, agent: AgentList, info: GroupInfo, ctx: SearchContext)->UpdateInfo:
        return cls(
            agent=agent, swarm=SwarmContext.of(info,[agent.index]), 
            search_context=ctx,
            random=Random(ctx.ndims),
            vars={}
        )   
        