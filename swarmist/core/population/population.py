from typing import Callable, List, Optional, Union
from pymonad.either import Either
import sys
import numpy as np
from ..dictionary import *
from ..errors import assert_equal_length, try_catch
from .agent import create_agent

def init_topology(agents: AgentList, builder: TopologyBuilder)->Topology:
    topology = builder(agents) if builder else None
    if topology and not callable(topology):
        assert_equal_length(len(topology), len(agents), "Number of neighborhoods")
        return lambda _: topology 
    return topology

def create_agents(pos_generator: PosGenerationMethod, size: int, ctx: SearchContext)->AgentList:
    return [
        create_agent(
            ctx=ctx,
            pos_generator=pos_generator,
            index=i
        ) for i in range(size)
    ]

def init_population(init: Initialization, ctx: SearchContext)->Either[Population, Exception]:
    def callback()->Population:
        agents = create_agents(init.generate_pos, init.population_size,ctx)
        topology = init_topology(agents, init.topology)
        return get_population(agents, topology)
    return try_catch(callback)

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

def get_population(agents: AgentList, topology: Optional[Callable[..., StaticTopology]] = None)->Population:
    return Population(
        agents=agents,
        topology=topology,
        size=len(agents)
    )

def get_population_rank(population: Population)->PopulationInfo:
    agents = population.agents
    topology = population.topology(agents) if population.topology else None 
    population_rank: GroupInfo = get_agents_rank(agents)
    groups:  List[GroupInfo] = []
    groups: List[GroupInfo] = (
        [population_rank for _ in agents] if not topology
        else [get_agents_rank([agents[i] for i in group]) for group in topology]
    )
    return PopulationInfo(
        info=population_rank, 
        group_info=groups
    )