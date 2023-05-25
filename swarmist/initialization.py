from __future__ import annotations
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from swarmist.core.dictionary import PosGenerationMethod, TopologyBuilder, StaticTopology, SearchContext, AgentList
from swarmist.core.random import BondedRandom

class InitializationMethods:

    def random(self)->PosGenerationMethod:
        return self.uniform()

    def uniform(self)->PosGenerationMethod:
        def callback(ctx: SearchContext)->PosGenerationMethod:
            return BondedRandom(
                lbound=ctx.bounds.min, ubound=ctx.bounds.max, size=ctx.ndims
            ).uniform()
        return callback

    def beta(self, alpha: float = 2.0, beta: float = 2.0)->PosGenerationMethod:
        def callback(ctx: SearchContext)->PosGenerationMethod:
            return BondedRandom(
                lbound=ctx.bounds.min, ubound=ctx.bounds.max, size=ctx.ndims
            ).beta(alpha=alpha, beta=beta)
        return callback
    
    def exponential(self, scale: float = 1.0)->PosGenerationMethod:
        def callback(ctx: SearchContext)->PosGenerationMethod:
            return BondedRandom(
                lbound=ctx.bounds.min, ubound=ctx.bounds.max, size=ctx.ndims
            ).exponential(scale=scale)
        return callback

    def rayleigh(self, scale: float = 1.0)->PosGenerationMethod:
        def callback(ctx: SearchContext)->PosGenerationMethod:
            return BondedRandom(
                lbound=ctx.bounds.min, ubound=ctx.bounds.max, size=ctx.ndims
            ).rayleigh(scale=scale)
        return callback

    def normal(self, loc: float = 0.0, scale: float = 1.0)->PosGenerationMethod:
        def callback(ctx: SearchContext)->PosGenerationMethod:
            return BondedRandom(
                lbound=ctx.bounds.min, ubound=ctx.bounds.max, size=ctx.ndims
            ).normal(loc=loc, scale=scale)
        return callback

    def lognormal(self, loc: float = 0.0, scale: float = 1.0)->PosGenerationMethod:
        def callback(ctx: SearchContext)->PosGenerationMethod:
            return BondedRandom(
                lbound=ctx.bounds.min, ubound=ctx.bounds.max, size=ctx.ndims
            ).lognormal(loc=loc, scale=scale)
        return callback
    
    def skewnormal(self, shape: float = 0.0, loc: float = 0.0, scale: float = 1.0)->PosGenerationMethod:
        def callback(ctx: SearchContext)->PosGenerationMethod:
            return BondedRandom(
                lbound=ctx.bounds.min, ubound=ctx.bounds.max, size=ctx.ndims
            ).skewnormal(shape=shape, loc=loc, scale=scale)
        return callback

    def weibull(self, shape:float = 1.0, scale:float = 1.0)->PosGenerationMethod:
        def callback(ctx: SearchContext)->PosGenerationMethod:
            return BondedRandom(
                lbound=ctx.bounds.min, ubound=ctx.bounds.max, size=ctx.ndims
            ).weibull(shape=shape, scale=scale)
        return callback

    def cauchy(self, loc:float = 0.0, scale:float = 1.0)->PosGenerationMethod:
        def callback(ctx: SearchContext)->PosGenerationMethod:
            return BondedRandom(
                lbound=ctx.bounds.min, ubound=ctx.bounds.max, size=ctx.ndims
            ).cauchy(loc=loc, scale=scale)
        return callback
 
    def levy(self, loc:float = 0.0, scale:float = 1.0)->PosGenerationMethod:
        def callback(ctx: SearchContext)->PosGenerationMethod:
            return BondedRandom(
                lbound=ctx.bounds.min, ubound=ctx.bounds.max, size=ctx.ndims
            ).levy(loc=loc, scale=scale)
        return callback
    
class TopologyMethods:
    def gbest()->TopologyBuilder:
        return lambda _: None

    def lbest(k:int = 2)->TopologyBuilder:
        def callback(agents: AgentList)->StaticTopology:
            n = len(agents)
            topology = []
            for i in range(n):
                neighbors = [i]
                for j in range(1, k):
                    neighbors.append((i + j) % n)
                    neighbors.append((i - j) % n)
                topology.append(neighbors)
            return topology
        return callback
