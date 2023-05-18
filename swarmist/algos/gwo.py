from __future__ import annotations
import numpy as np
from swarmist.core.dictionary import *
from swarmist.core.strategy import select, all
from swarmist.utils import random
from .helpers import *

class Gwo(UpdateMethodBuilder):
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
                method=self.update,
                where=where
            )
        ]
    
    def update(self,ctx: UpdateContext)->Agent: 
        ndims = ctx.ndims
        a = 2 - ctx.ctx.curr_gen * (2 / ctx.ctx.max_gen) 
        A1, A2, A3 = 2 * a * np.random.random() - a, 2 * a * np.random.random() - a, 2 * a * np.random.random() - a
        C1, C2, C3 = 2 * np.random.random(), 2 * np.random.random(), 2 * np.random.random()
        X1, X2, X3 = np.zeros(ndims), np.zeros(ndims), np.zeros(ndims)
        alpha, beta, gamma = ctx.best(3)
        pos = np.zeros(ndims)
        for j in range(ndims):
            X1[j] = alpha.best[j] - A1 * abs(C1 * alpha.best[j] - ctx.agent.best[j])
            X2[j] = beta.best[j] - A2 * abs(C2 *  beta.best[j] - ctx.agent.best[j])
            X3[j] = gamma.best[j] - A3 * abs(C3 * gamma.best[j] - ctx.agent.best[j])
            pos[j] = (X1[j] + X2[j] + X3[j])/3
        return self.recombination(ctx.agent, pos)
  