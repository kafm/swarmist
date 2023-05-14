from __future__ import annotations
from typing import Callable, Union
from dataclasses import replace
import numpy as np
from swarmist.core.dictionary import *
from swarmist.core.strategy import select, all, roulette
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
        reference: ReferenceGetter = self_pos(),
        xover_reference: ReferenceGetter = self_pos(),
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

    def pipeline(self)->UpdatePipeline:
        employees_selection = select(all())
        onlookers_selection = select(roulette()) 
        replace_selection = select() #TODO
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
        pm = self.centroid(ctx).average()
        ref = self.reference(ctx).average()
        diff = random.uniform(low=-1,high=1) * (ref - pm)
        xpos = self.xover_reference(ctx).average()
        return self.recombination(ctx.agent, xpos + diff)
        
        
    def replace(self, ctx: UpdateContext)->Agent:
        return get_new(ctx.agent, self.replace_pos(ctx))