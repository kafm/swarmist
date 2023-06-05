import swarmist as sw

expression = """
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X**2)
)
USING (
    PARAMETERS (
        A = 2 BOUNDED BY (0, 10)
    )
    POPULATION SIZE(40) INIT RANDOM_UNIFORM()
    SELECT ALL (
        UPDATE (
            A = PARAM(A) - CURR_GEN * ( PARAM(A) / MAX_GEN )
            SC = REPEAT(
                IF_THEN(
                    RANDOM(SIZE=1) < 0.5, 
                    SIN( RANDOM_UNIFORM(LOW=0, HIGH=2*PI(), SIZE=1) ),
                    COS( RANDOM_UNIFORM(LOW=0, HIGH=2*PI(), SIZE=1) )
                ), 
                NDIMS) 
            POS = POS + ( A * SC * ABS( RANDOM() * SWARM_BEST() - POS ) ) 
        ) WHEN IMPROVED = TRUE
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