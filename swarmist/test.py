import numpy as np
from core import *

search(
    space(
        dimensions(30),
        bounded(10, 20),
        constrained_by(lambda pos: pos),
        minimize(lambda x: x**2)
    ),
    until(
        max_evals=50000
    ),
    using(
        size(40),
        init(lambda ctx: [np.random.uniform(ctx.bounds.min, ctx.bounds.max, size=ctx.evaluate)]),
        None,
        parameters(
            ("self_confidence", 2.05)
        ),
        update(
            lambda ctx: None,
            select(all()),
            where(lambda a: a.improved)
        )
    )
)