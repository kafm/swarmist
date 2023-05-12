from __future__ import annotations
from typing import Callable
from dataclasses import replace
import numpy as np
from swarmist.core.dictionary import Pos, Agent

RecombinationMethod = Callable[[Agent, Pos], Agent]

def binomial(cr_probability: float = .6)->RecombinationMethod:
    def callback(agent: Agent, target: Pos)->Agent:
        n = agent.ndims
        r = np.random.randint(low=0, high = n, size = 1)
        new_pos = np.copy(agent.pos)
        for i in range(n):
            if np.random.uniform(0,1) < cr_probability or i == r:
                new_pos[i] = target[i]
        return replace(agent, pos=new_pos)
    return callback
  
def exponential(cr_probability: float = .6)->RecombinationMethod:
    def callback(agent: Agent, target: Pos)->Agent:
        n = agent.ndims
        j = n - 1
        new_pos = np.copy(agent.pos)
        for _ in range(j):
            if np.random.uniform(0,1) < cr_probability:
                i = r + 1 if r < j else 0
                new_pos[i] = target[i]
                r = i
            else:
                break
        return replace(agent, pos=new_pos)
    return callback

def replace_all()->RecombinationMethod:
    def callback(agent: Agent, target: Pos)->Agent:
        return replace(agent, pos=np.copy(target))
    return callback

def k_random(k:int = 1)->RecombinationMethod:
    def callback(agent: Agent, target: Pos)->Agent:
        n = agent.ndims
        r = np.random.randint(low=0, high = n, size = k)
        new_pos = np.copy(agent.pos)
        for i in r:
            new_pos[i] = target[i]
        return replace(agent, pos=new_pos)
    return callback