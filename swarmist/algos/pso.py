from __future__ import annotations
from typing import Callable, Union
from dataclasses import replace
import numpy as np
from swarmist.core.dictionary import *
from .helpers import *

class Pso(UpdateMethodBuilder):
    """
    Particle Swarm Optimization implementation based on the constriction coefficient proposal made by Clerc and Kennedy [1].
    In this implementation we use the Type 1''  constriction coefficient.

    [1] https://ieeexplore.ieee.org/abstract/document/985692
    """
    def __init__(self, 
        centroid: PosListGetter = global_best,
        reference: PosGetter = self_pos, #TODO check better the reference
        crossover_with: PosGetter = self_pos,
        recombination: RecombinationMethod = replace_all,
        c1: Union[float, Callable] = 2.05,
        c2: Union[float, Callable] = 2.05,
        chi: float = .729
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            crossover_with = crossover_with,
            recombination = recombination
        )
        self.c1 = c1 if callable(c1) else lambda: c1
        self.c2 = c2 if callable(c2) else lambda: c2
        self.chi = chi if callable(chi) else lambda: chi
    
    def update(self, ctx: UpdateContext)->Pos:
        ndims = ctx.agent.ndims
        ref = self.reference(ctx)
        centers = self.centroid(ctx)
        pm = np.sum(self.centroid(ctx), axis=0)/len(centers)
        velocity = self.chi() * (
                ctx.agent.delta +
                self.c1() * np.random.rand(ndims) * (ctx.agent.best - ref) +
                self.c2() * np.random.rand(ndims) * (pm - ref)
        )
        pos = self.recombination(ctx.agent, ctx.agent.pos + velocity)    
        return replace(ctx.agent, pos=pos)
    

class Fips(Pso):
    """
    The Fully Informed Particle Swarm is a variation of PSO proposed by Mendes, Kennedy and Neves [1].

    [1] https://ieeexplore.ieee.org/abstract/document/1304843
    """
    def __init__(self, 
        centroid: PosListGetter = stochastic_cog,
        reference: PosGetter = self_pos,
        crossover_with: PosGetter = self_pos,
        recombination: RecombinationMethod = replace_all,
        c1: Union[float, Callable] = 2.05,
        c2: Union[float, Callable] = 2.05,
        chi: float = .729
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            crossover_with = crossover_with,
            recombination = recombination,
            c1=c1,c2=c2,chi=chi
        )
    
    def update(self, ctx: UpdateContext)->UpdateMethod:
        c = (self.c1()+self.c2()) / ctx.size()
        ref = self.reference(ctx)
        pm =  self.centroid(ctx)
        persistence = ctx.agent.delta
        sct = np.random.rand(ctx.agent.ndims) * c * (pm - ref) #Check if * w is needed for Social central tendency
        pos = self.recombination(ctx.agent, self.crossover_with(ctx) + self.chi() * (persistence + sct) )
        return replace(ctx.agent, pos=pos)    
    
class Barebones(Pso):
    """
    Barebones Particle Swarms is a variation of PSO proposed by Kennedy [1].

    [1] https://ieeexplore.ieee.org/abstract/document/1202251
    """
    def __init__(self, 
        centroid: PosListGetter = global_best,
        reference: PosGetter = self_pos,
        crossover_with: PosGetter = self_pos,
        recombination: RecombinationMethod = replace_all
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            crossover_with = crossover_with,
            recombination = recombination
        )
    
    def update(self, ctx: UpdateContext)->UpdateMethod:
        centers = self.centroid(ctx)
        pm = np.sum(self.centroid(ctx), axis=0)/len(centers)
        ref = self.reference(ctx)
        mu = np.divide(np.add(ref, pm),2)
        sd = np.abs(np.subtract(ref, pm))
        pos = np.random.normal(mu, sd)
        return replace(ctx.agent, pos=pos)    

def get_chi(phi: float=4.1, k: float=.729) -> float:
    return np.sqrt(k) if phi <= 4 else np.sqrt(
        (2*k) / (phi - 2 + np.sqrt(phi**2 - 4 * phi)))