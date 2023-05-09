from typing import Callable, List, Optional, Union
from dataclasses import replace
import sys
import numpy as np
from .dictionary import AgentList, Agent, GroupInfo, StaticTopology, PopulationInfo, Population, UpdateContext, Fit, OneOrMoreAgents, T


def get_fits(agents: AgentList)->List[float]:
    return [a.fit for a in agents]

def fit_to_prob(fits: List[float])->List[int]:
    maxFit,minFit, total = (sys.float_info.max,-sys.float_info.max,0)
    for fit in fits:
        total += fit
        maxFit = max(maxFit, fit)
        minFit = min(minFit, fit)
    return [(maxFit - fit) / (maxFit - minFit)/total for fit in fits]

def get_best_k(agents:AgentList, rank:List[int], size: Optional[int] = None)->Union[Agent, List[Agent]]:
    if not size:
         return agents[rank[0]]
    return [agents[i] for i in rank[:size]]

def get_worse_k(agents:AgentList, rank:List[int], size: Optional[int] = None)->Union[Agent, List[Agent]]:
    if not size:
        return agents[rank[-1]]
    return [agents[i] for i in rank[-size:]]

def pick_random(
    agents: AgentList,  
    exclude: List[Agent]=None, 
    size: Optional[int] = None, 
    replace:bool=False, 
    p:List[float]= None
)->Union[Agent,List[Agent]]:
        return np.random.choice(
            agents if not exclude else filter(lambda a: a not in exclude, agents),
            size=size, replace=replace, p=p
        )

def get_agents_rank(agents: AgentList)->GroupInfo:
    fits = get_fits(agents)
    probs = fit_to_prob(fits)
    gsize = len(agents)
    rank = sorted(np.arange(gsize), key=lambda i: fits[i])
    return GroupInfo(
        all=lambda: agents,
        size=lambda: gsize,
        fits=lambda: fits,
        probs=lambda: probs,
        best=lambda size=None: get_best_k(agents, rank, size),
        worse=lambda size=None:  get_worse_k(agents, rank, size),
        filter=lambda f: filter(agents,f),
        map=lambda f: map(f, agents), 
        pick_random=lambda size=None, replace=False: pick_random(
            agents=agents,
            size=size, 
            replace=replace
        ), 
        pick_roulette=lambda size=None, replace=False: pick_random(
            agents=agents,
            size=size, 
            replace=replace,
            p=probs
        )
    )

def get_update_context(agent: Agent, info: GroupInfo, replace: bool = False)->UpdateContext:
    return UpdateContextWrapper(agent, info, replace).getContext()

def get_population(agents: AgentList, topology: Optional[Callable[..., StaticTopology]] = None)->Population:
    return Population(
        agents=agents,
        size=len(agents),
        rank=lambda: get_population_rank(agents, topology)
    )

def get_population_rank(agents: AgentList, topology: Optional[StaticTopology] = None)->PopulationInfo:
    population_rank: GroupInfo = get_agents_rank(agents)
    groups: List[GroupInfo] = (
        [population_rank for _ in agents] if not topology
        else [get_agents_rank([agents[i] for i in group]) for group in topology]
    )
    return PopulationInfo(
        info=population_rank, 
        group_info=groups
    )


class UpdateContextWrapper:
    def __init__(self, agent: Agent, info: GroupInfo, replace:bool=False):
        self.agent = agent
        self.info = info
        self.replace = replace
        self.picked: AgentList = []

    def getContext(self)->UpdateContext:
        return UpdateContext(
            agent=self.agent,
            all = self.info.all,
            size = self.info.size,
            fits = self.info.fits,
            probs = self.info.probs,
            filter = self.info.filter, 
            map = self.info.map,
            best = self._best,
            worse = self._worse,
            pick_random = self._pick_random,
            pick_roulette = self._pick_roulette
        )
    
    def _best(self, k: Optional[int] = None)->Union[Agent, AgentList]:
        return self.info.best(k)
    
    def _worse(self, k: Optional[int] = None)->Union[Agent, AgentList]:
        return self.info.worse(k)
    
    def _pick_random(self, k: Optional[int] = None)-> OneOrMoreAgents:
        return pick_random(self.info.all(),size=k) #TODO deal with exclude

    def _pick_roulette(self, k: Optional[int] = None)->OneOrMoreAgents:
        return pick_random(self.info.all(),size=k, p=self.info.probs) #TODO deal with exclude

    