from abc import ABC, abstractmethod
from typing import Callable, List
from swarmist.core.dictionary import UpdateContext, UpdateMethod, AgentList, Agent, Pos

PosListGetter = Callable[[UpdateContext], List[Pos]]
PosGetter = Callable[[UpdateContext], Pos]
RecombinationMethod = Callable[[Pos, Pos], Pos]

class UpdateMethodBuilder(ABC):
    def __init__(self,
        centroid: PosListGetter = None,
        reference: PosGetter = None,
        crossover_with: PosGetter = None,
        recombination: RecombinationMethod = None
    ):
        self.centroid = centroid
        self.reference = reference
        self.crossover_with = crossover_with
        self.recombination = recombination
        
    @abstractmethod
    def update(self, ctx: UpdateContext)->Agent:
        pass
