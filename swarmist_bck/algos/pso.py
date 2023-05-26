from __future__ import annotations
from typing import Callable, Union
from dataclasses import replace
import numpy as np
from swarmist_bck.core.dictionary import *
from swarmist_bck.core.strategy import select, all
from swarmist_bck.utils.random import Random
from .helpers import *

class Pso(UpdateMethodBuilder):
    """
    Particle Swarm Optimization implementation based on the constriction coefficient proposal made by Clerc and Kennedy [1].
    In this implementation we use the Type 1''  constriction coefficient.

    [1] https://ieeexplore.ieee.org/abstract/document/985692
    """
    def __init__(self, 
        centroid: ReferenceGetter = best_neighbor(),
        reference: ReferenceGetter = self_best(),
        xover_reference: ReferenceGetter = self_pos(),
        recombination: RecombinationMethod = replace_all(),
        c1: Union[float, Callable] = 2.05,
        c2: Union[float, Callable] = 2.05,
        chi: float = .729
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            xover_reference=xover_reference,
            recombination = recombination
        )
        self.c1 = c1 if callable(c1) else lambda: c1
        self.c2 = c2 if callable(c2) else lambda: c2
        self.chi = chi if callable(chi) else lambda: chi

    def pipeline(self)->UpdatePipeline:
        selection = select(all())
        return [
            lambda: Update(
                selection=selection(),
                method=self.update
            )
        ]
    
    def update(self, ctx: UpdateContext)->Agent:
        ndims = ctx.agent.ndims
        ref = self.reference(ctx).avg()
        pm = self.centroid(ctx).avg()
        pos = ctx.agent.pos
        velocity = self.chi() * (
                ctx.agent.delta +
                self.c1() * Random(ndims).rand() * (ref - pos) +
                self.c2() * Random(ndims).rand() * (pm - pos)
        )
        xpos = self.xover_reference(ctx).avg()
        return self.recombination(ctx.agent,  xpos + velocity)
    

class Fips(Pso):
    """
    The Fully Informed Particle Swarm is a variation of PSO proposed by Mendes, Kennedy and Neves [1].

    [1] https://ieeexplore.ieee.org/abstract/document/1304843
    """
    def __init__(self, 
        centroid: ReferenceGetter = all_neighbors(),
        reference: ReferenceGetter = self_pos(),
        recombination: RecombinationMethod = replace_all(),
        c1: Union[float, Callable] = 2.05,
        c2: Union[float, Callable] = 2.05,
        chi: float = .729
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            recombination = recombination,
            c1=c1,c2=c2,chi=chi
        )
     
    #TODO
    def update(self, ctx: UpdateContext)->Agent:
        ndims = ctx.agent.ndims
        centers = self.centroid(ctx).get()
        ref = self.reference(ctx).average()
        n = len(centers)
        phi = self.c1()+self.c2()
        c = phi / n
        pm = np.zeros(ndims)
        w = np.zeros(n)
        for p in centers:
            wi = random.rand(ndims)
            pm += wi * p
            w += wi
        pm = pm/w
        sct = random.rand(ndims) * c * w * (pm - ref)  # Social central tendency
        velocity = self.chi() * (ctx.agent.delta + sct)
        xpos = self.xover_reference(ctx).best().pos()
        return self.recombination(ctx.agent, xpos + velocity)
    
class Barebones(Pso):
    """
    Barebones Particle Swarms is a variation of PSO proposed by Kennedy [1].

    [1] https://ieeexplore.ieee.org/abstract/document/1202251
    """
    def __init__(self, 
        centroid: ReferenceGetter = best_neighbor(),
        reference: ReferenceGetter = self_best(),
        recombination: RecombinationMethod = replace_all()
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            recombination = recombination
        )
    
    def update(self, ctx: UpdateContext)->Agent:
        pm = self.centroid(ctx).average()
        ref = self.reference(ctx).average()
        mu = np.divide(np.add(ref, pm),2)
        sd = np.abs(np.subtract(ref, pm))
        return self.recombination(ctx.agent, random.normal(loc=mu, scale=sd)) 

def get_chi(phi: float=4.1, k: float=.729) -> float:
    return np.sqrt(k) if phi <= 4 else np.sqrt(
        (2*k) / (phi - 2 + np.sqrt(phi**2 - 4 * phi)))