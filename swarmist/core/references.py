from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from typing import List, Callable, Union, Optional
import numpy as np
from swarmist.core.dictionary import (
    Pos,
    Fit,
    AgentList,
    Agent,
    UpdateContext,
    IReference,
    IReferences,
)
from swarmist.core.random import Random


@dataclass(frozen=True)
class Reference(IReference):
    agent: Agent

    def is_better(self, other: Reference) -> bool:
        return self.agent.fit < other.agent.fit

    def get(self, key: Union[str, Callable[[Reference], Pos]] = "best") -> Pos:
        return self.agent[key] if isinstance(key, str) else key(self.agent)

    def add(
        self,
        other: Union[Reference, Pos, int, float],
        key: Union[str, Callable[[Reference], Pos]] = "best",
    ) -> Pos:
        return np.add(self.get(key=key), self._get_val(other))

    def subtract(
        self,
        other: Union[Reference, Pos, int, float],
        key: Union[str, Callable[[Reference], Pos]] = "best",
    ) -> Pos:
        return np.subtract(self.get(key=key), self._get_val(other))

    def multiply(
        self,
        other: Union[Reference, Pos, int, float],
        key: Union[str, Callable[[Reference], Pos]] = "best",
    ) -> Pos:
        return np.multiply(self.get(key=key), self._get_val(other))

    def divide(
        self,
        other: Union[Reference, Pos, int, float],
        key: Union[str, Callable[[Reference], Pos]] = "best",
    ) -> Pos:
        return np.divide(self.get(key=key), self._get_val(other))

    def power(
        self,
        other: Union[Reference, Pos, int, float],
        key: Union[str, Callable[[Reference], Pos]] = "best",
    ) -> Pos:
        return np.power(self.get(key=key), self._get_val(other))

    def modulus(
        self,
        other: Union[Reference, Pos, int, float],
        key: Union[str, Callable[[Reference], Pos]] = "best",
    ) -> Pos:
        return np.mod(self.get(key=key), self._get_val(other))

    def _get_val(self, val: Union[Reference, Pos, int, float]) -> Pos:
        return val.get() if isinstance(Reference, str) else val


@dataclass(frozen=True)
class References(IReferences):
    refs: List[Reference]

    @classmethod
    def of(agents: AgentList) -> References:
        return References([Reference(agent) for agent in agents])

    def get(self, index: int) -> Reference:
        return self.refs[index]

    def indices(self) -> List[int]:
        return [ref.agent.index for ref in self.refs]

    def pop(self) -> Reference:
        return self.refs.pop()

    def reduce(
        self, accumulator: Callable[[Reference], Pos], initial: Pos = None
    ) -> Pos:
        return reduce(accumulator, self.refs, initial)

    def sum(self, key: Union[str, Callable[[Reference], Pos]] = "best") -> Pos:
        callback = self._get_key_callback(key)
        return self.reduce(lambda a, v: v + callback(a), 0)

    def avg(
        self,
        weights: List[float] = None,
        key: Union[str, Callable[[Reference], Pos]] = "best",
    ) -> Pos:
        total = self.sum(key=key)
        return np.divide(total, self.size() if not weights else weights)

    def min(
        self, key: Union[str, Callable[[Reference], Union[Pos, Fit]]] = "fit"
    ) -> Pos:
        callback = self._get_key_callback(key)
        return min(self.refs, key=callback)

    def max(
        self, key: Union[str, Callable[[Reference], Union[Pos, Fit]]] = "fit"
    ) -> Pos:
        callback = self._get_key_callback(key)
        return max(self.refs, key=callback)

    def diff(
        self,
        other: Union[Reference, Pos, int, float],
        key: Union[str, Callable[[Reference], Pos]] = "best",
        reversed: bool = False,
    ) -> Pos:
        _other = other.get() if isinstance(other, Reference) else other
        key_callback = self._get_key_callback(key)
        callback = (
            lambda a: np.subtract(_other, key_callback(_other))
            if reversed
            else lambda a: np.subtract(key_callback(a), _other)
        )
        return self.reduce(callback, 0)

    def reverse_diff(
        self,
        other: Union[Reference, Pos, int, float],
        key: Union[str, Callable[[Reference], Pos]] = "best",
    ) -> Pos:
        return self.diff(other, key, reversed=True)

    def _get_key_callback(
        self, key: Union[str, Callable[[Reference], Pos]]
    ) -> Callable[[Agent], Union[Pos, Fit]]:
        return lambda a: getattr(a, key) if isinstance(key, str) else key

    def size(self) -> int:
        return len(self.refs)


class SwarmMethods:
    def best(self, size: Optional[int] = None) -> Callable[[UpdateContext], Reference]:
        if not size:
            return lambda ctx: ctx.swarm.best()
        return lambda ctx: ctx.swarm.k_best(size)

    def worse(self, size: Optional[int] = None) -> Callable[[UpdateContext], Reference]:
        if not size:
            return lambda ctx: ctx.swarm.worse()
        return lambda ctx: ctx.swarm.k_worse(size)

    def all(self) -> Callable[[UpdateContext], References]:
        return self.neighborhood()

    def neighborhood(self) -> Callable[[UpdateContext], References]:
        return lambda ctx: ctx.swarm.all()

    def rand_to_best(self, f: float = 0.5) -> Callable[[UpdateContext], Pos]:
        def callback(ctx: UpdateContext) -> Pos:
            best = ctx.swarm.best().get(key="best")
            a = ctx.swarm.pick_random_unique().get(key="best")
            return f * best + (1 - f) * a

        return callback

    def current_to_best(self, f: float = 0.5) -> Callable[[UpdateContext], Pos]:
        def callback(ctx: UpdateContext) -> Pos:
            best = ctx.swarm.best().get(key="best")
            current = ctx.agent.best
            return current + f * (best - current)

        return callback

    def pick_random(
        self, size: Optional[int] = None, replace: bool = False, unique: bool = False
    ) -> Callable[[UpdateContext], Union[Reference, References]]:
        if unique:
            return lambda ctx: ctx.swarm.pick_random_unique(size, replace=replace)
        return lambda ctx: ctx.swarm.pick_random(size, replace=replace)

    def pick_roulette(
        self, size: Optional[int] = None, replace: bool = False, unique: bool = False
    ) -> Callable[[UpdateContext], Reference]:
        if unique:
            return lambda ctx: ctx.swarm.pick_roulette_unique(size, replace=replace)
        return lambda ctx: ctx.swarm.pick_roulette(size, replace=replace)


class AgentMethods:
    def index(self) -> Callable[[Union[UpdateContext, Agent]], int]:
        return lambda ctx: ctx.agent.index if not isinstance(ctx, Agent) else ctx.index

    def ndims(self) -> Callable[[Union[UpdateContext, Agent]], int]:
        return lambda ctx: ctx.agent.ndims if not isinstance(ctx, Agent) else ctx.ndims

    def delta(self) -> Callable[[Union[UpdateContext, Agent]], Pos]:
        return lambda ctx: ctx.agent.delta if not isinstance(ctx, Agent) else ctx.delta

    def trials(self) -> Callable[[Union[UpdateContext, Agent]], int]:
        return (
            lambda ctx: ctx.agent.trials if not isinstance(ctx, Agent) else ctx.trials
        )

    def improved(self) -> Callable[[Union[UpdateContext, Agent]], bool]:
        return (
            lambda ctx: ctx.agent.improved
            if not isinstance(ctx, Agent)
            else ctx.improved
        )

    def best(self) -> Callable[[Union[UpdateContext, Agent]], Pos]:
        return lambda ctx: ctx.agent.best if not isinstance(ctx, Agent) else ctx.best

    def pos(self) -> Callable[[Union[UpdateContext, Agent]], Pos]:
        return lambda ctx: ctx.agent.pos if not isinstance(ctx, Agent) else ctx.pos

    def fit(self) -> Callable[[Union[UpdateContext, Agent]], Fit]:
        return lambda ctx: ctx.agent.fit if not isinstance(ctx, Agent) else ctx.fit
