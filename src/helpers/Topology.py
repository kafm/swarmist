from __future__ import annotations
from typing import List, Optional
from .Individual import Individual, Neighborhood


def  buildTopology(individuals: List[Individual], topologyName:str, neighborhoodRange:Optional[int]=None)->List[Neighborhood]:
    if topologyName == "lbest":
        return LBEST(individuals, 2 if neighborhoodRange == None else neighborhoodRange)
    else: 
        return GBEST(individuals)
        
def GBEST(individuals: List[Individual])->List[Neighborhood]:
    topology = []
    neighborhood: Neighborhood = Neighborhood(0)
    for i in individuals: 
        neighborhood.append(i)
        topology.append(neighborhood)
    return topology

def LBEST(individuals: List[Individual], k:int=2)->List[Neighborhood]:
    topology:List[Neighborhood] = []
    n = len(individuals)
    for i in range(n): 
        neighborhood:Neighborhood = Neighborhood(i)
        for j in range(k):
            neighborhood.append(individuals[(i + j) % n])
            neighborhood.append(individuals[(i - j) % n])
        topology.append(neighborhood)
    return topology

    
