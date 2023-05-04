from typing import Callable, List, Optional, Union
from functools import reduce
import sys
import numpy as np
from .dictionary import AgentList, Agent, GroupInfo, StaticTopology, PopulationInfo, Population


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
    rank = sorted(np.arange(gsize), key=lambda i: probs[i])
    return GroupInfo(
        all=lambda: agents,
        size=lambda: gsize,
        fits=lambda: fits,
        probs=lambda: probs,
        best=lambda size: get_best_k(agents, rank, size),
        worse=lambda size:  get_worse_k(agents, rank, size),
        filter=lambda f: filter(agents,f),
        map=lambda f: map(f, agents), 
        reduce=lambda f: reduce(f, agents),
        pick_random=lambda exclude, size, replace: pick_random(
            agents=agents,
            exclude=exclude,
            size=size, 
            replace=replace
        ), 
        pick_roulette=lambda exclude, size, replace: pick_random(
            agents=agents,
            exclude=exclude,
            size=size, 
            replace=replace,
            p=probs
        )
    )


def get_population(agents: AgentList, topology: Optional[Callable[..., StaticTopology]] = None)->Population:
    return Population(
        agents=agents,
        size=len(agents),
        rank=lambda: get_population_rank(agents, topology)
    )

def get_population_rank(agents: AgentList, topology: Optional[StaticTopology])->PopulationInfo:
    population_rank: GroupInfo = get_agents_rank(agents)
    groups: List[GroupInfo] = (
        [population_rank for _ in agents] if not topology
        else [get_agents_rank([agents[i] for i in group]) for group in topology]
    )
    return PopulationInfo(
        info=population_rank, 
        group_info=groups
    )
