from typing import cast
from swarmist.core.dictionary import *
from swarmist.utils import random
import numpy as np

@dataclass(frozen=True)
class Reference:  
    average: Callable[..., Pos]
    get: List[Pos]

def best_neighbor()->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        pos = ctx.best(1)[0].best
        return Reference(
            average=lambda: pos,
            get=lambda: [pos]
        )
    return callback

def k_best_neighbors(size: int= 1)->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        pos_list = [a.best for a in ctx.best(size)]
        return Reference(
            average=average_pos(pos_list, size),
            get=lambda: pos_list
        )
    return callback

def worse_neighbor()->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        pos = ctx.worse(1)[0].best
        return Reference(
            average=lambda: pos,
            get=lambda: [pos]
        )
    return callback

def k_worse_neighbors(size: int= 1)->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        pos_list = [a.best for a in ctx.worse(size)]
        return Reference(
            average=average_pos(pos_list, size),
            get=lambda: pos_list
        )
    return callback

def all_neighbors()->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        pos_list = [a.best for a in  ctx.all()]
        return Reference(
            average=average_pos(pos_list, ctx.size()),
            get=lambda: pos_list
        )
    return callback

def pick_random(size: int = 1, replace:bool = False)->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        pos_list = ctx.pick_random(size, replace=replace)
        return Reference(
            average=average_pos(pos_list, ctx.size()),
            get=lambda: pos_list
        )
    return callback

def pick_random_unique(size: int = 1, replace:bool = False)->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        pos_list = ctx.pick_random_unique(size, replace=replace)
        return Reference(
            average=average_pos(pos_list, ctx.size()),
            get=lambda: pos_list
        )
    return callback

def pick_roulette(size: int = 1, replace:bool = False)->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        pos_list = ctx.pick_roulette(size, replace=replace)
        return Reference(
            average=average_pos(pos_list, ctx.size()),
            get=lambda: pos_list
        )
    return callback

def pick_roulette_unique(size: int = 1, replace:bool = False)->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        pos_list = ctx.pick_roulette_unique(size, replace=replace)
        return Reference(
            average=average_pos(pos_list, ctx.size()),
            get=lambda: pos_list
        )
    return callback

def self_best()->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        pos = ctx.agent.best
        return Reference(
            average=lambda: pos,
            get=lambda: [pos]
        )
    return callback

def self_pos()->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        pos = ctx.agent.pos
        return Reference(
            average=lambda: pos,
            get=lambda: [pos]
        )
    return callback

def random_pos()->Callable[[UpdateContext], Reference]:
    def callback(ctx: UpdateContext)->Reference:
        ndims = ctx.agent.ndims
        pos = random.uniform(low=ctx.bounds.min, high=ctx.bounds.max, size=ndims)
        return Reference(
            average=lambda: pos,
            get=lambda: [pos]
        )
    return callback

# def stochastic_cog(ctx: UpdateContext)->List[Pos]:
#     ndims = ctx.agent.ndims
#     w = np.zeros(ndims)
#     pm = np.zeros(ndims)
#     neighbors = ctx.all()
#     for neighbor in neighbors:
#         wi = np.random.rand(ndims)
#         w += wi
#         pm += np.multiply(wi, neighbor.best)
#     return [np.divide(pm, w)]

#TODO 
# def rand_to_best(ctx: UpdateContext, y: int, f: float, gamma: float)->List[Pos]:
#     best = ctx.info.best()
#     choices: AgentList = ctx.info.pick_random(exclude=[ctx.agent, best])
#     a = gamma * best.pos + (1 - gamma) * a.pos  
#     return get_y_diff(choices, y * 2, f)

# def currentToBestUpdate(agent: Agent, neighbors: Neighborhood, crMethod: CrossoverMethod, y: int, f: float):
#     best: Agent = neighbors.best()
#     nY = y * 2
#     choices: List[Agent] = neighbors.getRandomIndividuals(k=nY,excludeIndexes=[agent.index, best.index])
#     a = agent.pos + f * (best.pos - agent.pos)
#     diff = getYDiff(choices, nY , f)
#     pos = crMethod(
#         agent.pos, 
#         lambda i : a[i] + diff[i]
#     )
#     updatePosIfBetterFit(agent, pos)

# def get_y_diff(agents: List[Agent], n_y: int, f: float) -> List[float]:
#     middlePoint = int(len(agents)/n_y)
#     return f * sum([
#         agents[i].pos - agents[i+middlePoint].pos 
#         for i in range(middlePoint)
#     ]) 


def average_pos(pos_list: List[Pos], size: int)->Pos:
    return np.sum(pos_list)/size