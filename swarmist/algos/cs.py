from __future__ import annotations
from typing import Callable, Union, cast
from dataclasses import replace
import numpy as np
import math
from swarmist.core.dictionary import *
from swarmist.core.strategy import select, all, with_probability
from swarmist.utils import random
from .helpers import *

class Cs(UpdateMethodBuilder):
    """
    Cuckoo Search  proposed by Yang and Deb [1]. Implemented as suggested in [2].

    [1] https://ieeexplore.ieee.org/abstract/document/5393690
    [2] https://link.springer.com/chapter/10.1007/978-3-319-02141-6_1
    """
    def __init__(self, 
        centroid: ReferenceGetter = best_neighbor(),
        reference: ReferenceGetter = self_pos(),
        local_centroid: ReferenceGetter = pick_random(),
        local_reference: ReferenceGetter = pick_random(),
        global_recombination: RecombinationMethod = replace_all(),
        local_recombination: RecombinationMethod = k_with_probability(.75),
        alpha: float = 1,
        mu: float = 1.5 #replaced lambda with mu due to reserved python lambda 
    ):
        super().__init__(
            centroid = centroid,
            reference = reference
        )
        self.local_centroid = local_centroid
        self.local_reference = local_reference
        self.global_recombination = global_recombination
        self.local_recombination = local_recombination 
        self.alpha = alpha if callable(alpha) else lambda: alpha
        self.mu = mu if callable(mu) else lambda: mu
       
    def pipeline(self)->UpdatePipeline:
        global_update_selection = select(all())
        local_update_selection = select(all()) #select(with_probability())
        return [
            lambda: Update(
                selection=global_update_selection(),
                method=self.global_update,
                where=lambda a: a.improved
            ), 
            lambda: Update(
                selection=local_update_selection(),
                method=self.local_update,
                where=lambda a: a.improved
            )
        ]
    
    def global_update(self, ctx: UpdateContext)->Agent:
        ndims = ctx.ndims
        pos = ctx.agent.pos
        best = self.centroid(ctx).sum()
        ref = self.reference(ctx).sum()
        pos = pos + self.alpha() *random.levy2(loc=self.mu(), size=ndims) * np.subtract(ref, best)
        return self.global_recombination(ctx.agent, pos)
    
    def local_update(self, ctx: UpdateContext)->Agent:
        ndims = ctx.ndims
        a = self.local_centroid(ctx).sum()
        b = self.local_reference(ctx).sum()
        pos = ctx.agent.pos + (np.random.rand(ndims) * np.subtract(a, b))
        return self.local_recombination(ctx.agent, pos)