
from functools import reduce
from operator import add
import numpy as np
 
# initializing list
lis = [1, 3, 5, 6, 2]

print(reduce(add, map(lambda x: x - 2, lis)))
res = 0
for el in lis:
    res += el - 2
print(res)

rands = np.random.rand(3, 4)
sum_i = column_sum = np.sum(rands, axis=0)
print(rands)
print(sum_i)


ndims = 3
n = 2
w = np.random.rand(n, ndims)
lis2 = [[1,2,3], [4,5,6]]
print(f"w={w}")
print(np.multiply(map(lambda x: x, lis),w),np.sum(w, axis=0))