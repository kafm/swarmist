from swarmist.core.dictionary import *
import numpy as np

#centroid
def global_best(ctx: UpdateContext)->List[Pos]:
    return [ctx.best().best]

def k_best(ctx: UpdateContext, size: Optional[int]= None)->List[Pos]: 
    return [a.best for a in ctx.best(size if size else ctx.info.size())]

def stochastic_cog(ctx: UpdateContext)->List[Pos]:
    ndims = ctx.agent.ndims
    w = np.zeros(ndims)
    pm = np.zeros(ndims)
    neighbors = ctx.all()
    for neighbor in neighbors:
        wi = np.random.rand(ndims)
        w += wi
        pm += np.multiply(wi, neighbor.best)
    return [np.divide(pm, w)]

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

#reference
def self_best(ctx: UpdateContext)->Pos:
    return ctx.agent.best

def self_pos(ctx: UpdateContext)->Pos:
    return ctx.agent.pos



