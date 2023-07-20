import swarmist as sw

problem, bounds = sw.benchmark.sphere()
numDimensions = 20
maxGenerations = 1000

strategy_expr = """
        PARAM  A = 2
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                A = PARAM(A) - 2 * CURR_GEN / MAX_GEN
                POS = AVG(
                        MAP(
                            SWARM_BEST(3), (REF) => REF.POS
                                - (A * (2 * RANDOM(SIZE=NDIMS) - 1))
                                * ABS((2 * RANDOM(SIZE=NDIMS)) * REF.POS - POS)
                        )
                    )
            ) WHEN IMPROVED = TRUE
        )   
"""
 
st = sw.sdl.strategy(strategy_expr) 

res = sw.search(
    sw.minimize(problem, bounds, dimensions=numDimensions),
    sw.until(max_gen=maxGenerations),
    sw.using(st),
)
print(res)
print(f"res={res[-1].fit}")