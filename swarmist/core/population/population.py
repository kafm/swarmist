from typing import Callable, List, Optional, Union
from pymonad.either import Either
import sys
import numpy as np
from ..dictionary import *
from ..errors import assert_equal_length, try_catch
from .agent import create_agent, get_fits, fit_to_prob

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

def get_best_k(agents:AgentList, rank:List[int], size: Optional[int] = 1)->List[Agent]:
    return [agents[i] for i in rank[:size]]

def get_worse_k(agents:AgentList, rank:List[int], size: Optional[int] = 1)->List[Agent]:
    return [agents[i] for i in rank[-size:]]

def pick_random(
    agents: AgentList,  
    size: Optional[int] = None, 
    replace:bool=False, 
    p:List[float]= None
)->Union[Agent,List[Agent]]:
        return np.random.choice(agents, size=size, replace=replace, p=p)

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
        best=lambda size=1: get_best_k(agents, rank, size),
        worse=lambda size=1:  get_worse_k(agents, rank, size),
        filter=lambda f: filter(f,agents),
        map=lambda f: map(f, agents), 
        pick_random=lambda size=1, replace=False: pick_random(
            agents=agents,
            size=size, 
            replace=replace
        ), 
        pick_roulette=lambda size=1, replace=True: pick_random(
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