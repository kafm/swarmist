from abc import ABC, abstractmethod
from typing import Callable, List
from swarmist_bck.core.dictionary import Update, UpdateContext, Pos, Agent
from .references import Reference
from .recombination import RecombinationMethod

ReferenceGetter = Callable[[UpdateContext], Reference]
RecombinationMethod = Callable[[Agent, Pos], Pos]
UpdatePipeline = List[Callable[..., Update]]

class UpdateMethodBuilder(ABC):
    def __init__(self,
        centroid: ReferenceGetter = None,
        reference: ReferenceGetter = None,
        xover_reference: ReferenceGetter = None,
        recombination: RecombinationMethod = None
        
    ):
        self.centroid = centroid
        self.reference = reference
        self.xover_reference = xover_reference
        self.recombination = recombination

    @abstractmethod
    def pipeline(self)->UpdatePipeline:
        pass
