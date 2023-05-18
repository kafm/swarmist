from __future__ import annotations
from typing import Callable, Union
from dataclasses import replace
import numpy as np
from swarmist.core.dictionary import *
from swarmist.core.strategy import select, all
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
        xover_reference: ReferenceGetter = self_best(),
        recombination: RecombinationMethod = replace_all()
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            xover_reference = xover_reference,
            recombination = recombination
        )

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
        ndims = ctx.agent.ndims
        pm = self.centroid(ctx).sum()
        ref = self.reference(ctx).sum()
        abs_pos = np.abs(ctx.agent.pos)
        diff = random.rand(ndims)*(pm - abs_pos) - random.rand(ndims)*(ref - abs_pos)
        xpos = self.xover_reference(ctx).best().pos()
        return self.recombination(ctx.agent, xpos + diff)


    