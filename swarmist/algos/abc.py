from __future__ import annotations
from typing import Callable, Union
from dataclasses import replace
import numpy as np
import sys
from swarmist.core.dictionary import *
from swarmist.core.strategy import select, all, roulette, biggest, when
from swarmist.utils import random
from .helpers import *

class Abc(UpdateMethodBuilder):
    """
    Artificial Bee Colony proposed by Karaboga[1]. Implemented as suggested in [1,2].

    [1] https://abc.erciyes.edu.tr/pub/tr06_2005.pdf
    [2] https://link.springer.com/article/10.1007/s10898-007-9149-x
    """
    def __init__(self, 
        centroid: ReferenceGetter = pick_random_unique(),
        reference: ReferenceGetter = self_best(),
        xover_reference: ReferenceGetter = self_best(),
        recombination: RecombinationMethod = k_random(k=1), 
        replace_pos: ReferenceGetter = random_pos(), 
        limit: int = None
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            xover_reference = xover_reference,
            recombination = recombination
        ) 
        self.replace_pos = replace_pos
        self.limit = limit

    def _select_to_replace(self, info: GroupInfo)->AgentList:
        agents: AgentList = info.all()
        n = info.size()
        to_replace = [a for a in agents if a.trials >= self.max_trials(n, a.ndims)]
        return to_replace

    def pipeline(self)->UpdatePipeline:
        employees_selection = select(all())
        onlookers_selection = select(roulette()) 
        replace_selection = select(
            biggest(
                key=lambda a: a.trials,
                selection=self._select_to_replace
            )
        )
        return [
            lambda: Update(
                selection=employees_selection(),
                method=self.update,
                where=lambda a: a.improved
            ),
            lambda: Update(
                selection=onlookers_selection(),
                method=self.update,
                where=lambda a: a.improved
            ),
            lambda: Update(
                selection=replace_selection(),
                method=self.replace
            ),
        ]
    
    def update(self,ctx: UpdateContext)->Agent: 
        ndims = ctx.agent.ndims
        pm = self.centroid(ctx).average()
        ref = self.reference(ctx).average()
        diff = np.random.uniform(low=-1,high=1, size=ndims) * (ref - pm)
        xpos = self.xover_reference(ctx).average()
        return self.recombination(ctx.agent, xpos + diff)
        
    def replace(self, ctx: UpdateContext)->Agent:
        pos = np.random.uniform(low=ctx.bounds.min, high=ctx.bounds.max, size=ctx.agent.ndims)
        print(f"Replacing {ctx.agent.index}")
        return replace(ctx.agent, 
            pos=pos,
            best=pos,
            delta=np.zeros(ctx.agent.ndims),
            fit=sys.float_info.max,
            trials=0
        )
    
    def max_trials(self, population_size: int, ndims: int)->int:
        return int(population_size/2)*ndims if not self.limit else self.limit
       
    