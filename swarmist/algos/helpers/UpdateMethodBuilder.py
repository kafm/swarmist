from abc import ABC, abstractmethod
from typing import Callable, List
from swarmist.core.dictionary import UpdateContext, Agent, Pos
from .references import Reference
from .recombination import RecombinationMethod

ReferenceGetter = Callable[[UpdateContext], Reference]
RecombinationMethod = Callable[[Pos, Pos], Pos]

class UpdateMethodBuilder(ABC):
    def __init__(self,
        centroid: ReferenceGetter = None,
        reference: ReferenceGetter = None,
        recombination: RecombinationMethod = None
    ):
        self.centroid = centroid
        self.reference = reference
        self.recombination = recombination
        
    @abstractmethod
    def update(self, ctx: UpdateContext)->Agent:
        pass
