from __future__ import annotations
from typing import Callable, Union
from dataclasses import replace
import numpy as np
from swarmist.core.dictionary import *
from swarmist.core.strategy import select, all
from swarmist.utils import random
from .helpers import *

class De(UpdateMethodBuilder):
    """
    Differential Evolution implementation proposed by Storn and Price [1].

    [1] https://link.springer.com/article/10.1023/A:1008202821328
    """
    def __init__(self, 
        centroid: ReferenceGetter = pick_random_unique(),
        reference: ReferenceGetter = pick_random_unique(),
        xover_reference: ReferenceGetter = pick_random_unique(),
        recombination: RecombinationMethod = binomial(),
        f: Union[float, Callable] = .5
    ):
        super().__init__(
            centroid = centroid,
            reference = reference,
            xover_reference=xover_reference,
            recombination = recombination
        )
        self.f = f if callable(f) else lambda: f
       
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
        a = self.xover_reference(ctx).best().pos()
        b = self.centroid(ctx).sum()
        c = self.reference(ctx).sum()
        pos = a + self.f() * (b-c)
        return self.recombination(ctx.agent, pos)