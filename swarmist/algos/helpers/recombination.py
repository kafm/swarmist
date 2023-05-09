from __future__ import annotations
from typing import Optional, List,  Tuple, cast, Any, Callable
import numpy as np
from swarmist.core.dictionary import Pos

def binomial(pos:Pos, cr_pos: Pos, cr: float)->Pos:
    n = len(pos)
    r = np.random.randint(low=0, high = n, size = 1)
    new_pos = np.copy(pos)
    for i in range(n):
        if np.random.uniform(0,1) < cr or i == r:
            new_pos[i] = cr_pos[i]
    return new_pos

def exponential(pos: Pos, cr_pos: Pos, cr: float)->Pos:
    n = len(pos)
    j = n - 1
    new_pos = np.copy(pos)
    for _ in range(j):
        if np.random.uniform(0,1) < cr:
            i = r + 1 if r < j else 0
            new_pos[i] = cr_pos[i]
            r = i
        else:
            break
    return new_pos

def replace_all(_, cr_pos: Pos)->Pos:
    return np.copy(cr_pos)

def k_random(pos: Pos, cr_pos: Pos, k:int = 1)->Pos:
    n = len(pos)
    r = np.random.randint(low=0, high = n, size = k)
    new_pos = np.copy(pos)
    for i in r:
        new_pos[i] = cr_pos[i]
    return new_pos