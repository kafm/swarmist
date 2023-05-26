from __future__ import annotations
from typing import List
from functools import partial, reduce
from dataclasses import replace
import numpy as np
from swarmist.core.dictionary import (
    Agent,
    AgentList,
    Topology,
    TopologyBuilder,
    PosGenerator,
    SearchStrategy,
    SearchContext,
    Condition,
    Recombination,
    PosEditor,
    PopulationInfo,
    GroupInfo,
    Update,
)
from swarmist.core.errors import assert_equal_length
from swarmist.core.info import AgentsInfo, UpdateInfo


class Population:
    def __init__(self, strategy: SearchStrategy, ctx: SearchContext):
        self._strategy = strategy
        self._size = strategy.initialization.population_size
        self._ndims = ctx.ndims
        self._bounds = ctx.bounds
        self._evaluate = ctx.evaluate
        self._agents = [
            self._create_agent(
                pos_generator=strategy.initialization.generate_pos, index=i, ctx=ctx
            )
            for i in range(self._size)
        ]
        self._topology = self._get_topology(strategy.initialization.topology)
        self._pipeline = strategy.update_pipeline
        self._ranking = self.rank(ctx)

    def rank(self, ctx: SearchContext) -> PopulationInfo:
        topology = self._topology(self._agents) if self._topology else None
        ranking = AgentsInfo.of(self._agents, ctx.bounds, ctx.ndims)
        groups: List[GroupInfo] = (
            [ranking for _ in self._agents]
            if not topology
            else [
                AgentsInfo.of([self._agents[i] for i in group], ctx.bounds, ctx.ndims)
                for group in topology
            ]
        )
        self._ranking = PopulationInfo(info=ranking, group_info=groups)
        return self._ranking

    def ranking(self) -> PopulationInfo:
        return self._ranking

    def update(self, ctx: SearchContext):
        for update in self._pipeline:
            to_update = update.selection(self._ranking.info)
            for agent in to_update:
                new_agent = self._get_updated_agent(agent, update, ctx)
                self._agents[new_agent.index] = new_agent
            self.rank(ctx)

    def _get_updated_agent(
        self, agent: Agent, update: Update, ctx: SearchContext
    ) -> Agent:
        info = UpdateInfo.of(agent, self._ranking.group_info[agent.index], ctx=ctx)
        return self._evaluate_and_get(
            agent=update.recombination(agent, update.editor(info)),
            old_agent=agent,
            condition=update.condition,
        )

    def _evaluate_and_get(
        self, agent: Agent, old_agent: Agent, condition: Condition
    ) -> Agent:
        pos, fit = self._evaluate(agent.pos)
        delta = pos - old_agent.pos
        improved = fit < agent.fit
        best = pos
        trials = 0
        if not improved:
            best = old_agent.best
            fit = old_agent.fit
            trials = old_agent.trials + 1
        new_agent = replace(
            agent,
            delta=delta,
            pos=pos,
            fit=fit,
            best=best,
            trials=trials,
            improved=improved,
        )
        return new_agent if not condition or condition(new_agent) else old_agent

    def _create_agent(
        self, pos_generator: PosGenerator, index: int, ctx: SearchContext
    ) -> Agent:
        ndims = ctx.ndims
        pos, fit = ctx.evaluate(pos_generator(ctx))
        return Agent(
            index=index,
            ndims=ndims,
            pos=pos,
            best=pos,
            delta=np.zeros(ndims),
            fit=fit,
            trials=0,
            improved=True,
        )

    def _get_topology(self, builder: TopologyBuilder) -> Topology:
        topology = builder(self._agents) if builder else None
        if topology and not callable(topology):
            assert_equal_length(
                len(topology), len(self._agents), "Number of neighborhoods"
            )
            return lambda _: topology
        return topology
