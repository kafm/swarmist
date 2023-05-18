from typing import Callable, List, Optional
from dataclasses import replace
from pymonad.either import Either
import numpy as np
from .dictionary import *
from .errors import assert_equal_length, try_catch
from .executor import SearchExecutor
from .info import AgentsInfo, UpdateInfo

def create_agent(pos_generator: PosGenerationMethod, index: int, ctx: SearchContext)->Agent:
    pos = ctx.clip(pos_generator(ctx))
    fit = ctx.evaluate(pos)  
    return Agent(
        index=index,
        ndims=ctx.ndims,
        pos=pos,
        best=pos,
        delta=np.zeros(ctx.ndims),
        fit=fit,
        trials=0,
        improved=True
    )

def do_apply(agent: Agent, old_agent: Agent, executor: SearchExecutor)->Agent: 
    #TODO check if other algorithms works like this
    pos = executor.clip(agent.pos)
    delta = pos - old_agent.pos 
    fit = executor.evaluate(pos)
    improved = fit < agent.fit
    best = pos 
    trials = 0
    if not improved:
        best = old_agent.best
        fit = old_agent.fit
        trials = old_agent.trials + 1 
        #print(f"agent {new_agent.index} did not improve")
    return replace(
        agent, 
        delta=delta,
        pos=pos,
        fit=fit,
        best=best,
        trials=trials,
        improved=improved
    )

def update_agent(update: Update, agent: Agent, info: GroupInfo, executor: SearchExecutor)->Agent:    
    where = update.where 
    new_agent = update.method(UpdateInfo.of(agent, info, executor.context()))
    candidate = do_apply(
        agent=new_agent, 
        old_agent=agent,
        executor=executor
    )
    return candidate if not where or where(candidate) else replace(agent, trials=candidate.trials)

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
        return Population(
            agents=agents, topology=topology, size=len(agents),
            ndims=ctx.ndims, bounds=ctx.bounds
        )
    return try_catch(callback)

def get_population_rank(population: Population)->PopulationInfo:
    agents = population.agents
    ndims = population.ndims
    bounds = population.bounds
    topology = population.topology(agents) if population.topology else None 
    population_rank = AgentsInfo.of(agents, bounds, ndims)
    groups:  List[GroupInfo] = []
    groups: List[GroupInfo] = (
        [population_rank for _ in agents] if not topology
        else [
            AgentsInfo.of([agents[i] for i in group],bounds, ndims)
            for group in topology
        ]
    )
    return PopulationInfo(
        info=population_rank, 
        group_info=groups
    )