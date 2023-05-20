from __future__ import annotations
from typing import Callable, Union
from dataclasses import replace
import numpy as np
from swarmist.core.dictionary import *
from swarmist.core.strategy import select, all
from swarmist.utils import random
from .helpers import *

class Sca(UpdateMethodBuilder):
    """
    Sine Cosine algorithm implementation proposed by Mirjalili [1].

    [1] https://www.sciencedirect.com/science/article/pii/S0950705115005043
    """
    def __init__(self, 
        centroid: ReferenceGetter = best_neighbor(),
        reference: ReferenceGetter = self_pos(),
        recombination: RecombinationMethod = replace_all(),
        a: float = 2
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            recombination = recombination
        )
        self.a = a if callable(a) else lambda: a
       
    def pipeline(self)->UpdatePipeline:
        selection = select(all())
        return [
            lambda: Update(
                selection=selection(),
                method=self.update,
                where=lambda a: a.improved
            )
        ]
    
    def update(self, ctx: UpdateContext)->Agent:
        ndims = ctx.ndims
        a = self.a()
        r1 = a - ctx.ctx.curr_gen * (a / ctx.ctx.max_gen) 
        r2 = np.random.uniform(low=0, high=2*np.pi, size=ndims)
        r3 = np.random.uniform(low=-2, high=2, size=ndims)
        r4 = np.random.uniform(size=ndims)
        pos = self.reference(ctx).sum()
        center = self.centroid(ctx).sum()
        for i in range(ndims): 
            sc = np.sin(r2[i]) if r4[i] < .5 else np.cos(r2[i])
            pos[i] += ( 
                r1 * 
                sc * 
                abs(r3[i] * center[i] - pos[i]) 
            )
        return self.recombination(ctx.agent, pos)