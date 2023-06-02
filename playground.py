from typing import List
from collections import namedtuple
#from swarmist_bck.core.dictionary import Agent
from dataclasses import fields, dataclass
from functools import reduce
from swarmist.core.references import Reference
from swarmist.core.dictionary import Agent
import numpy as np

a = np.array([])
b = np.array([4, 5])
print(np.concatenate((a, b)))
j = a+b
print(j[-1])

#agent_fields = {field.name: field.type for field in fields(Agent)}
#print(agent_fields)
# arr = np.array([[1,2, 4],[2,3, 6],[4,5, 10]])
# print(np.average(arr, axis=0))
# print(np.sum(arr, axis=0))

# @dataclass(frozen=True)
# class Test:
#     els: List[int]


# test = Test([1,2])
# print(test.els)
# test.els.append(3)
# test.els.extend([4,5])
# print(test.els)
# print(test.els[:1])

# r = np.random.uniform(size=20)
# print(r)
# print(np.linalg.norm(r))                                               
# print(np.abs(r))

# n = 20
# dim = 10
# pa = .25
# print(np.random.uniform(0, 1, (n, dim)) > pa)



#build a python compose function where output of one function is input of the next using reduce 
# from functools import reduce
# def compose(*funcs):
#     return reduce(lambda f, g: lambda x: f(g(x)), funcs, lambda x: x)
#

# vals =[1,2,3,4,5]

# print(reduce(function=lambda x, y: x + y, sequence=vals, initial=0))

# def test(**kargs):
#     print(kargs.pop("b"))
#     for key, value in kargs.items():
#         print(key, value)

# test(a=1, b=2, c=3)

# @dataclass(frozen=True)
# class SAgent(Agent):
#     def __getitem__(self, item):
#         return getattr(self, item)
    
# agent = SAgent(
#     pos=[1,2,3],
#     fit=1,
#     best=[1,2,3],
#     trials=0,
#     index= 0, 
#     ndims=3, 
#     delta=[0,0,0],
#     improved= False
# )

# reference = Reference(agent)

# print(agent["pos"])
# print(reference.get("pos"))