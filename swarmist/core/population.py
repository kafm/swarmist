
from __future__ import annotations
from typing import List
from dataclasses import replace
import numpy as np
from swarmist.core.dictionary import Agent, AgentList, Topology, TopologyBuilder, Update, PosGenerator, SearchStrategy, SearchContext, Condition, Recombination, PosEditor, PopulationInfo, GroupInfo, Initialization
from swarmist.core.errors import assert_equal_length
from swarmist.core.info import AgentsInfo, UpdateInfo

class Population:
    def __init__(self, 
        strategy: SearchStrategy, 
        ctx: SearchContext):
        self.ctx = ctx
        self.strategy = strategy
        self.size = strategy.initialization.population_size
        self.agents = [
            self._create_agent(
                ctx=ctx,
                pos_generator=strategy.initialization.generate_pos,
                index=i
            ) for i in range(self.size)
        ]
        self.topology = self._get_topology(self.agents, strategy.initialization.topology)
        self.pipeline = strategy.update_pipeline
    
    def update(self, ctx: SearchContext, _population: Population = None, index=0)->Population:
        population = _population if _population else self
        if index == len(self.pipeline):
            return population
        update = self.pipeline[index]
        rank = self._get_rank(population)
        to_update:AgentList = update.selection(rank.info)
        new_agents:AgentList = population.agents.copy()
        for agent in to_update:
            new_agents[agent.index] = self._update_agent(
                agent=agent,
                info=UpdateInfo.of(agent, rank.group_info[agent.index], ctx),
                pos_editor=update.editor,
                recombinator=update.recombination,
                condition=update.condition
            )
        return self.update(replace(self, agents=new_agents), index+1)
    
    def _get_rank(self, population: Population)->PopulationInfo:
        topology = self.topology(population.agents) if self.topology else None 
        agents = population.agents
        population_rank = AgentsInfo.of(agents, self.ctx.bounds, self.ctx.ndims)
        groups: List[GroupInfo] = (
            [population_rank for _ in agents] if not topology
            else [
                AgentsInfo.of([agents[i] for i in group], self.ctx.bounds, self.ctx.ndims)
                for group in topology
            ]
        )
        return PopulationInfo(
            info=population_rank, 
            group_info=groups
        )

    def _update_agent(self, 
        agent: Agent, 
        info: UpdateInfo,
        pos_editor: PosEditor, 
        recombinator: Recombination,
        condition: Condition
    )->Agent:
        new_agent = self._evaluate_and_get(
                recombinator(agent, pos_editor(info),
                agent             
            )
        )
        if not condition or condition(new_agent):
            return new_agent
        return agent

    def _evaluate_and_get(self, agent: Agent, old_agent: Agent)->Agent: 
        pos, fit = self.ctx.evaluate(agent.pos)
        delta = pos - old_agent.pos 
        improved = fit < agent.fit
        best = pos 
        trials = 0
        if not improved:
            best = old_agent.best
            fit = old_agent.fit
            trials = agent.trials + 1    
        return replace(
            agent, 
            delta=delta,
            pos=pos,
            fit=fit,
            best=best,
            trials=trials,
            improved=improved
        )


    def _create_agent(pos_generator: PosGenerator, index: int, ctx: SearchContext)->Agent:
        pos, fit = ctx.evaluate(pos)  
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
    
    def _get_topology(agents: AgentList, builder: TopologyBuilder)->Topology:
        topology = builder(agents) if builder else None
        if topology and not callable(topology):
            assert_equal_length(len(topology), len(agents), "Number of neighborhoods")
            return lambda _: topology 
        return topology