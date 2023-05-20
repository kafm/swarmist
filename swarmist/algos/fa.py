from __future__ import annotations
from typing import Callable, Union
from dataclasses import replace
import numpy as np
from swarmist.core.dictionary import *
from swarmist.core.strategy import select, all
from swarmist.utils import random
from .helpers import *

class Fa(UpdateMethodBuilder):
    """
    Firefly implementation proposed by Yang [1]. Delta parameter included as suggested in [2].

    [1] https://link.springer.com/chapter/10.1007/978-3-642-04944-6_14
    [2] https://link.springer.com/chapter/10.1007/978-3-319-02141-6_1
    """
    def __init__(self, 
        centroid: ReferenceGetter = all_neighbors(),
        recombination: RecombinationMethod = replace_all(),
        alpha: float = 1,
        beta: float = 1,
        gamma:float = .01,
        delta: float = .97
    ):
        super().__init__(
            centroid = centroid,
            recombination = recombination
        )
        self.alpha = alpha if callable(alpha) else lambda: alpha
        self.beta = alpha if callable(beta) else lambda: beta
        self.gamma = gamma if callable(gamma) else lambda: gamma
        self.delta = delta if callable(delta) else lambda: delta
       
    def pipeline(self)->UpdatePipeline:
        selection = select(all())
        return [
            lambda: Update(
                selection=selection(),
                method=self.update
            )
        ]

    def _get_pos_accumulator(self, ctx: UpdateContext, alpha: float)->Callable[[Pos], Pos]:
        def callback(agent: Agent, pos: Pos)->Pos:
            if agent.fit < ctx.agent.fit:
                d = agent.pos - pos
                ffBeta = self.beta() * np.exp(-self.gamma() *  np.square(d))
                e = alpha * (np.random.random(ctx.ndims) - 0.5)
                return pos + ffBeta * d + e
            return pos
        return callback

    def update(self, ctx: UpdateContext)->Agent: 
        alpha = (self.alpha()*self.delta())**ctx.ctx.curr_gen
        pos = np.copy(ctx.agent.pos)
        pos = self.centroid(ctx).reduce(
            self._get_pos_accumulator(ctx, alpha),
            pos
        )
        return self.recombination(ctx.agent, pos)