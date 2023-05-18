from __future__ import annotations
from typing import Optional, List, Callable
from dataclasses import dataclass
from .dictionary import Agent, AgentList, Bounds, Fit, GroupInfo, UpdateContext, SearchContext
import numpy as np

def get_fits(agents: AgentList)->List[float]:
    return [a.fit for a in agents]

def fit_to_prob(fits: List[float])->List[int]:
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
    
    def best(self, k: int = 1)->AgentList:
        return [self.agents[i] for i in self.rank[:k]]

    def worse(self, k: int = 1)->AgentList:
        return [self.agents[i] for i in self.rank[-k:]]

    def filter(self, f: Callable[[Agent], bool])->AgentList: 
        return filter(f,self.agents)

    def pick_random(self, k: int = 1, replace: bool = False)->AgentList:
        return np.random.choice(self.agents, size=k, replace=replace) 
        
    def pick_roulette(self, k: int = 1, replace: bool = False)->AgentList:
        return np.random.choice(self.agents, size=k, replace=replace, p=self.probs)
        
@dataclass(frozen=True)
class UpdateInfo(UpdateContext):
    info: GroupInfo
    picked: List[int]

    @classmethod
    def of(cls, agent: AgentList, info: GroupInfo, ctx: SearchContext)->UpdateInfo:
        return cls(
            agent=agent, info=info, picked=[agent.index],
            bounds=info.bounds, ndims=info.ndims, ctx=ctx
        )

    def all(self)->AgentList:
        return self.info.all()

    def size(self)->int: 
        return self.info.size()
    
    def filter(self, f: Callable[[Agent], bool])->AgentList: 
        return filter(f,self.info.all())

    def best(self, k: int = 1)->AgentList:
        agents = self.info.best(k)
        self._append_picked(agents)
        return agents

    def worse(self, k: int = 1)->AgentList:
        agents =self.info.worse(k)
        self._append_picked(agents)
        return agents

    def pick_random(self, k: int = 1, replace: bool = False)->AgentList:
        agents = self.info.pick_random(replace=replace,size=k)
        self._append_picked(agents)
        return agents
        
    def pick_roulette(self, k: int = 1, replace: bool = False)->AgentList:
        agents = self.info.pick_random(size=k, replace=replace, p=self.info.probs)
        self._append_picked(agents)
        return agents
        
    def pick_random_unique(self, k: Optional[int] = None, replace: Optional[bool] = False)->AgentList:
        agents = list(self.info.filter(lambda a: a.index not in self.picked))
        agents = np.random.choice(agents, size=k, replace=replace)
        self._append_picked(agents)
        return agents
    
    def pick_roulette_unique(self, k: Optional[int] = None, replace: Optional[bool] = True)->AgentList:
        agents = self.info.filter(lambda a: a.index not in self.picked)
        agents = np.random.choice(
            agents, size=k, replace=replace, 
            p=fit_to_prob(get_fits(agents))
        )
        self._append_picked(agents)

    def _append_picked(self, agents: AgentList):
        self.picked.extend(agents)