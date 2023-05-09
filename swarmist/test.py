import numpy as np
from core import *
from algos.pso import Pso

res = search(
    search_space=space(
        dimensions=dimensions(30),
        bounds=bounded(10, 20),
        constraints=constrained_by(lambda pos: pos ),
        fit_function=minimize(lambda x: sum(x**2))
    ),
    stop_condition=until(
        max_evals=50000
    ),
    search_strategy=using(
        size(40),
        init(lambda ctx: np.random.uniform(ctx.bounds.min, ctx.bounds.max, size=ctx.ndims)),
        None,
        update(
            select(all()),
            apply(Pso().update)
            #apply(lambda ctx: ctx.agent),
            #where(lambda a: a.improved)
        )
    ), 
    replace=False
)

print(res)