from __future__ import annotations
from typing import Optional, cast
from functools import reduce
from Dictionary import *
from Exceptions import SearchEnded
import numpy as np
import sys

def get_fits(agents: AgentList[Agent])->List[float]:
    return [a.fit for a in agents]

def fit_to_prob(fits: List[float])->List[int]:
    maxFit,minFit, total = (sys.float_info.max,-sys.float_info.max,0)
    for fit in fits:
        total += fit
        maxFit = max(maxFit, fit)
        minFit = min(minFit, fit)
    return [(maxFit - fit) / (maxFit - minFit)/total for fit in fits]

def get_best_k(agents:AgentList[Agent], rank:List[int], size: Optional[int] = None)->Union[Agent, List[Agent]]:
    if not size:
         return agents[rank[0]]
    return [agents[i] for i in rank[:size]]

def get_worse_k(agents:AgentList[Agent], rank:List[int], size: Optional[int] = None)->Union[Agent, List[Agent]]:
    if not size:
        return agents[rank[-1]]
    return [agents[i] for i in rank[-size:]]

def pick_random(
    agents: AgentList[Agent],  
    exclude: List[Agent]=None, 
    size: Optional[int] = None, 
    replace:bool=False, 
    p:List[float]= None
)->Union[Agent,List[Agent]]:
        return np.random.choice(
            agents if not exclude else filter(lambda a: a not in exclude, agents),
            size=size, replace=replace, p=p
        )


def get_agents_rank(agents: AgentList[Agent])->GroupInfo:
    fits = get_fits(agents)
    probs = fit_to_prob(fits)
    size = len(agents)
    rank = sorted(np.arange(size), key=lambda i: probs[i])
    return GroupInfo(
        all=lambda: agents,
        size=lambda: size,
        fits=lambda: fits,
        props=lambda: probs,
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




