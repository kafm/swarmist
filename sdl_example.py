import swarmist as sw

expression = """
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X ^ 2)
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
            ALPHA = ( PARAM(ALPHA) * PARAM(DELTA) ) ^ CURR_GEN
            VALUES = MAP(ALL(), (REF) => IF_THEN(REF.FIT < FIT, REF.POS, 0))
            POS = REDUCE(
                VALUES, 
                (ACC, VAL) => ACC + (
                    ( PARAM(BETA) * EXP( -1 * PARAM(GAMMA) * (VAL - ACC)^2 ))
                    * (VAL - ACC) + ALPHA * RANDOM_UNIFORM(LOW=-1, HIGH=1)
                ),
                POS
            )
        )
    )
)
UNTIL (
    GENERATION = 1000
)
"""

            # BETTER_ONES = FILTER(ALL(), (REF) => REF.FIT < FIT)
            # POS = REDUCE(
            #     BETTER_ONES, 
            #     (ACC, REF) => ACC + ( 
            #             PARAM(BETA) * EXP( -1 * PARAM(GAMMA) * (REF - POS)^2 
            #         ) * (REF - POS) + ALPHA * RANDOM_UNIFORM(LOW=-1, HIGH=1)
            #     ),
            #     POS


results = sw.sdl.execute(expression) 