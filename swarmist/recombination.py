from __future__ import annotations
from typing import Callable, Union
from dataclasses import replace
import numpy as np
from swarmist.core.dictionary import Pos, Agent, UpdateContext

RecombinationMethod = Callable[[Agent, Pos], Agent]


class RecombinationMethods:
    def binomial(
        self, cr_probability: Union[float, Callable[[UpdateContext], float]] = 0.6
    ) -> RecombinationMethod:
        p = cr_probability if callable(cr_probability) else lambda _: cr_probability

        def callback(ctx: UpdateContext, target: Pos) -> Agent:
            agent: Agent = ctx.agent
            n = agent.ndims
            r = np.random.randint(low=0, high=n)
            new_pos = np.copy(agent.pos)
            cr = p(ctx)
            for i in range(n):
                if np.random.uniform(0, 1) < cr or i == r:
                    new_pos[i] = target[i]
            return replace(agent, pos=new_pos)

        return callback

    def exponential(
        self, cr_probability: Union[float, Callable[[UpdateContext], float]] = 0.6
    ) -> RecombinationMethod:
        p = cr_probability if callable(cr_probability) else lambda _: cr_probability

        def callback(ctx: UpdateContext, target: Pos) -> Agent:
            agent: Agent = ctx.agent
            n = agent.ndims
            j = n - 1
            r = np.random.randint(low=0, high=n)
            new_pos = np.copy(agent.pos)
            cr = p(ctx)
            for _ in range(j):
                if np.random.uniform(0, 1) < cr:
                    i = r + 1 if r < j else 0
                    new_pos[i] = target[i]
                    r = i
                else:
                    break
            return replace(agent, pos=new_pos)

        return callback

    def k_with_probability(
        self, probability: Union[float, Callable[[UpdateContext], float]] = 0.25
    ) -> RecombinationMethod:
        p = probability if callable(probability) else lambda _: probability

        def callback(ctx: UpdateContext, target: Pos) -> Agent:
            agent: Agent = ctx.agent
            n = agent.ndims
            new_pos = np.copy(agent.pos)
            prob = p(ctx)
            for i in range(n):
                if np.random.uniform(0, 1) < prob:
                    new_pos[i] = target[i]
            return replace(agent, pos=new_pos)

        return callback

    def replace_all(self) -> RecombinationMethod:
        def callback(ctx: UpdateContext, target: Pos) -> Agent:
            return replace(ctx.agent, pos=np.copy(target))

        return callback

    def get_new(self) -> RecombinationMethod:
        def callback(ctx: UpdateContext, target: Pos) -> Agent:
            agent: Agent = ctx.agent
            pos = np.copy(target)
            return replace(agent, pos=pos, best=pos, fit=np.inf, trials=0)

        return callback

    def k_random(
        self, k: Union[float, Callable[[UpdateContext], int]] = 1
    ) -> RecombinationMethod:
        size = k if callable(k) else lambda _: k

        def callback(ctx: UpdateContext, target: Pos) -> Agent:
            agent: Agent = ctx.agent
            n = agent.ndims
            r = np.random.randint(low=0, high=n, size=size(ctx))
            new_pos = np.copy(agent.pos)
            for i in r:
                new_pos[i] = target[i]
            return replace(agent, pos=new_pos)

        return callback
