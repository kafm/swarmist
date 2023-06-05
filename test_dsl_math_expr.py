import swarmist as sw

expression = """
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X**2)
)
USING (
    PARAMETERS (
        ALPHA = 1 BOUNDED BY (0, 2)
        DELTA = 0.97 BOUNDED BY (0, 1)
        BETA = 1 BOUNDED BY (0, 2)
        GAMMA = 0.01 BOUNDED BY (0, 1)
    )
    POPULATION SIZE(40) INIT RANDOM_UNIFORM()
    SELECT ALL (
        UPDATE (
            ALPHA = ( PARAM(ALPHA) * PARAM(DELTA) ) ** CURR_GEN
            BETTER_ONES = FILTER(ALL(), (REF) => REF.FIT < FIT)
            POS = REDUCE(BETTER_ONES, (ACC, REF) => 
                ACC + (
                    PARAM(BETA) * EXP( -PARAM(GAMMA) * (REF.POS - POS)**2 )
                ) * (REF.POS - POS) + ( ALPHA * (RANDOM() - 0.5) )
            , POS)
        )
    )
)
UNTIL (
    GENERATION = 1000
)
"""
#POS = POS + ( A * SC * ABS( RANDOM() * DIFF(SWARM_BEST(), POS) ) ) 
                # ffBeta = self.beta() * np.exp(-self.gamma() *  np.square(agent.pos - pos))
                # e = alpha * (np.random.random(ctx.ndims) - 0.5)
                # return pos + ffBeta * (agent.pos - pos) + e


# pos = ctx.agent.pos + (np.random.rand(ndims) * np.subtract(a, b))
#SCT = RANDOM() * (PHI/N) * W * AVG(MAP(NEIGHBORS, (REF) => REF - POS), W)
#.format(bounds=bounds)
results = sw.sdl.execute(expression) 