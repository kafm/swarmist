from typing import List
from collections import namedtuple
from swarmist.core.dictionary import Agent
from dataclasses import fields, dataclass
import numpy as np

#agent_fields = {field.name: field.type for field in fields(Agent)}
#print(agent_fields)
# arr = np.array([[1,2, 4],[2,3, 6],[4,5, 10]])
# print(np.average(arr, axis=0))
# print(np.sum(arr, axis=0))

@dataclass(frozen=True)
class Test:
    els: List[int]


test = Test([1,2])
print(test.els)
test.els.append(3)
test.els.extend([4,5])
print(test.els)
print(test.els[:1])