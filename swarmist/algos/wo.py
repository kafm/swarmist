from __future__ import annotations
from functools import partial
import numpy as np
from swarmist.core.dictionary import *
from swarmist.core.strategy import select, all
from swarmist.utils import random
from .helpers import *

class Wo(UpdateMethodBuilder):
    """
    Whale Optimization Algorithm implementation proposed by Mirjalili and Lewis [1].

    [1] https://www.sciencedirect.com/science/article/pii/S0965997816300163
    """
    def __init__(self, 
        centroid: ReferenceGetter = k_best_neighbors(3),
        reference: ReferenceGetter = self_best(),
        explore_reference: ReferenceGetter = pick_random_unique(),
        recombination: RecombinationMethod = replace_all(),
        spiral: float = .5
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            recombination = recombination
        ) 
        self.explore_reference = explore_reference
        self.spiral = spiral if callable(spiral) else lambda: spiral

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
        A = np.full(ndims, 2 * a * np.random.random() - a)
        pos:Pos = if_then(
            lambda: np.random.uniform() < .5,
            lambda: if_then(
                lambda: abs(A[0]) < 1,
                lambda: self.encircle(ctx, A),
                lambda: self.explore(ctx, A)    
            ),
            lambda: self.attack(ctx)
        )
        return self.recombination(ctx.agent, pos)
 
    def encircle(self, ctx: UpdateContext, A: List[float])->List[float]:
        C =  2 * np.random.random(size=ctx.ndims)
        best = self.centroid(ctx).sum()
        pos = self.reference(ctx).sum()
        D = abs( C * best - pos )
        return best - A * D

    def attack(self, ctx: UpdateContext)->List[float]:
        best = self.centroid(ctx).sum()
        pos = self.reference(ctx).sum()
        D = abs(best - pos)
        l = np.random.uniform(-1.0, 1.0, size=ctx.ndims)
        return D * np.exp(self.spiral()*l) * np.cos(2.0*np.pi*l) + best

    def explore(self, ctx: UpdateContext, A: List[float]):
        exp = self.explore_reference(ctx).sum()
        pos = self.explore_reference(ctx).sum()
        C = 2 * np.random.random(size=ctx.ndims)
        D = abs( C * exp - pos )    
        return exp - A * D