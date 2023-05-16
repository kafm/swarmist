
from functools import reduce
from operator import add
import numpy as np




def average_pos(pos_lis, size):
    return np.divide(sum(pos_lis),size)

def sum_pos(pos_list): 
    return sum(pos_list)

size = 20
pos = np.random.uniform(low=0, high=10,size=size)
pos2 = np.random.uniform(low=0, high=10,size=size)
print(f"pos={pos}, pos2={pos2},avg={average_pos([pos, pos2], 2)}, sum={sum_pos([pos, pos2])}")