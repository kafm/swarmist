import swarmist as sw

problem, bounds = sw.benchmark.sphere()
numDimensions = 20
maxGenerations = 1000

strategy_expr = """
    PARAMETERS (
        ALPHA = 1 BOUNDED BY (0, 2)
        DELTA = 0.97 BOUNDED BY (0, 1)
        BETA = 1 BOUNDED BY (0, 2)
        GAMMA = 0.01 BOUNDED BY (0, 1)
    )
    POPULATION SIZE(40) INIT RANDOM_UNIFORM()
    SELECT ALL (
        UPDATE (
            ALPHA = ( PARAM(ALPHA) * PARAM(DELTA) ) **  CURR_GEN
            VALUES = MAP(ALL(), (REF) => IF_THEN(REF.FIT < FIT, REF.POS, 0))
            POS = REDUCE(
                VALUES, 
                (ACC, VAL) => ACC + (
                    ( PARAM(BETA) * EXP( -1 * PARAM(GAMMA) * (VAL - ACC)** 2 ))
                    * (VAL - ACC) + ALPHA * RANDOM_UNIFORM(LOW=-1, HIGH=1)
                ),
                POS
            )
        )
    )
"""


st = sw.sdl.strategy(strategy_expr) 

res = sw.search(
    sw.minimize(problem, bounds, dimensions=numDimensions),
    sw.until(max_gen=maxGenerations),
    sw.using(st),
)
print(f"res={min(res, key=lambda x: x.fit)}")