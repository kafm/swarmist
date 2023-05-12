from __future__ import annotations
from typing import Callable, Union
from dataclasses import replace
import numpy as np
from swarmist.core.dictionary import *
from swarmist.utils import random
from .helpers import *

class Jaya(UpdateMethodBuilder):
    """
    Jaya implementation proposed by Rao [1].

    [1] http://growingscience.com/beta/ijiec/2072-jaya-a-simple-and-new-optimization-algorithm-for-solving-constrained-and-unconstrained-optimization-problems.html
    """
    def __init__(self, 
        centroid: ReferenceGetter = best_neighbor(),
        reference: ReferenceGetter = worse_neighbor(),
        recombination: RecombinationMethod = replace_all()
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            recombination = recombination
        )

    def update(self, ctx: UpdateContext)->Agent:
        ndims = ctx.agent.ndims
        pm = self.centroid(ctx).average()
        ref = self.reference(ctx).average()
        pos = ctx.agent.best
        abs_pos = np.abs(pos)
        diff = random.rand(ndims)*(pm - abs_pos) - random.rand(ndims)*(ref - abs_pos)
        return self.recombination(ctx.agent, pos + diff)
    