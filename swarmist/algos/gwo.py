from __future__ import annotations
from functools import partial
import numpy as np
from swarmist.core.dictionary import *
from swarmist.core.strategy import select, all
from swarmist.utils import random
from .helpers import *

class Gwo(UpdateMethodBuilder):
    """
    Grey Wolf Optimizer implementation proposed by Mirjalili, Mohammad Mirjalili and Lewis [1].

    [1] https://www.sciencedirect.com/science/article/pii/S0965997813001853
    """
    def __init__(self, 
        centroid: ReferenceGetter = k_best_neighbors(3),
        reference: ReferenceGetter = self_best(),
        recombination: RecombinationMethod = replace_all()
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            recombination = recombination
        ) 

    def pipeline(self)->UpdatePipeline:
        where = lambda a: a.improved
        selection = select(all())
        return [
            lambda: Update(
                selection=selection(),
                method=self.update,
                where=where
            )
        ]
    
    def _get_pos_transformer(self,ctx: UpdateContext)->Callable[[Pos], Pos]:
        def callback(center: Pos, reference: Pos, a: float)->Pos:
            A = 2 * a * np.random.random() - a
            C = 2 * np.random.random()
            return center - A * abs( C * center - reference)
        a = 2 - ctx.ctx.curr_gen * (2 / ctx.ctx.max_gen) 
        ref = self.reference(ctx).avg()
        return partial(callback, reference=ref, a=a)
    
    def update(self,ctx: UpdateContext)->Agent:
        pos = self.centroid(ctx).avg(self._get_pos_transformer(ctx))
        return self.recombination(ctx.agent, pos)
  