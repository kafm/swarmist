import random
from oslash import Left, Right

def rand(max_evals): 
  def iter_helper():
    curr_evals = 0
    while True:
        curr_evals += 1
        if not curr_evals or curr_evals > max_evals:
            raise Exception("Finished")
        yield (curr_evals,random.randint(1,101))
  iter = iter_helper() 
  return lambda: next(iter)
    


g = rand(9)
for _ in range(10):
    print(g())