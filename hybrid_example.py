import swarmist as sw

problem, bounds = sw.benchmark.sphere()
numDimensions = 20
maxGenerations = 1000

strategy_expr = """
        PARAM POP_SIZE = AUTO INT BOUNDED BY (10, 100)
        PARAM NUM_TRIALS = AUTO INT BOUNDED BY (10, 300)
        POPULATION SIZE(PARAM(POP_SIZE)) INIT RANDOM_UNIFORM()
        SELECT ALL (
            USING RANDOM RECOMBINATION SIZE(1)
            UPDATE (
                POS = RANDOM_UNIFORM(LOW=-1, HIGH=1) * (POS - PICK_RANDOM())
            ) WHEN IMPROVED = TRUE
        )
        SELECT SIZE(1) WHERE TRIALS > PARAM(NUM_TRIALS) ORDER BY TRIALS DESC (
            INIT RANDOM_UNIFORM()
        )
"""


st = sw.sdl.strategy(strategy_expr) 

res = sw.search(
    sw.minimize(problem, bounds, dimensions=numDimensions),
    sw.until(max_gen=maxGenerations),
    sw.tune(st, max_gen=1),
)
print(res)
print(f"res={res.fit}")