from __future__ import annotations
import numpy as np
from swarmist.core.dictionary import *
from swarmist.core.strategy import select, all
from swarmist.utils import random
from .helpers import *

class Tlbo(UpdateMethodBuilder):
    """
    Teaching–learning-based optimization implementation proposed by Rao, Savsani and Vakharia [1].

    [1] https://www.sciencedirect.com/science/article/pii/S0010448510002484
    """
    def __init__(self, 
        centroid: ReferenceGetter = best_neighbor(),
        reference: ReferenceGetter = all_neighbors(),
        xover_reference: ReferenceGetter = self_best(),
        recombination: RecombinationMethod = replace_all(),
        partner_reference: ReferenceGetter = pick_random_unique(1),
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            xover_reference = xover_reference,
            recombination = recombination
        ) 
        self.partner_reference = partner_reference

    def pipeline(self)->UpdatePipeline:
        where = lambda a: a.improved
        selection = select(all())
        return [
            lambda: Update(
                selection=selection(),
                method=self.teacher_phase,
                where=where
            ),
            lambda: Update(
                selection=selection(),
                method=self.student_phase,
                where=where
            )
        ]
    
    def teacher_phase(self,ctx: UpdateContext)->Agent: 
        ndims = ctx.agent.ndims
        tf = np.random.choice([1,2])
        best = self.centroid(ctx).avg()
        mu = self.reference(ctx).avg()
        r = np.random.uniform(size=ndims)
        pos = ctx.agent.pos + (r * (best - tf*mu))
        return self.recombination(ctx.agent, pos)
        
    def student_phase(self, ctx: UpdateContext)->Agent:
        ndims = ctx.agent.ndims
        partner = self.partner_reference(ctx).best_agent()
        pos = ctx.agent.best 
        diff = (
            pos - partner.best  if ctx.agent.fit < partner.fit
            else partner.best - pos
        )
        r = np.random.uniform(size=ndims)
        return self.recombination(ctx.agent, pos + r * diff)