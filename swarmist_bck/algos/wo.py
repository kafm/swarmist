from __future__ import annotations
from functools import partial
import numpy as np
from swarmist_bck.core.dictionary import *
from swarmist_bck.core.strategy import select, all
from swarmist_bck.utils import random
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
        a: float = 2,
        spiral: float = .5
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            recombination = recombination
        ) 
        self.explore_reference = explore_reference
        self.a = a if callable(a) else lambda: a
        self.spiral = spiral if callable(spiral) else lambda: spiral

    def pipeline(self)->UpdatePipeline:
        selection = select(all())
        return [
            lambda: Update(
                selection=selection(),
                method=self.update
            )
        ]
        
    def update(self,ctx: UpdateContext)->Agent:
        ndims = ctx.ndims
        a = self.a()
        a = a - ctx.ctx.curr_gen * (a / ctx.ctx.max_gen) 
        A = ( 2 * np.multiply(a, np.random.random(size=ndims)) ) - a
        pos = if_then(
            lambda: np.random.uniform() < .5,
            lambda: if_then(
                lambda: np.linalg.norm(A) < 1,
                lambda: self.encircle(ctx, A),
                lambda: self.explore(ctx, A)    
            ),
            lambda: self.attack(ctx)
        )
        return self.recombination(ctx.agent, pos)
 
    def encircle(self, ctx: UpdateContext, A: List[float])->Pos:
        C =  2 * np.random.random(size=ctx.ndims)
        best = self.centroid(ctx).sum()
        pos = self.reference(ctx).sum()
        D = np.abs( C * best - pos )
        return best - A * D

    def attack(self, ctx: UpdateContext)->Pos:
        best = self.centroid(ctx).sum()
        pos = self.reference(ctx).sum()
        D = np.abs(best - pos)
        l = np.random.uniform(-1.0, 1.0, size=ctx.ndims)
        return D * np.exp(self.spiral()*l) * np.cos(2.0*np.pi*l) + best

    def explore(self, ctx: UpdateContext, A: List[float])->Pos:
        exp = self.explore_reference(ctx).sum()
        pos = self.explore_reference(ctx).sum()
        C = 2 * np.random.random(size=ctx.ndims)
        D = np.abs( C * exp - pos )  
        return exp - A * D