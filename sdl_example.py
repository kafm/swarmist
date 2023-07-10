import swarmist as sw


expression = """
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X **  2)
)
USING (
    PARAM ALPHA = AUTO FLOAT BOUNDED BY (0, 2) 
    PARAM BETA = AUTO FLOAT BOUNDED BY (0, 2) 
    PARAM DELTA = AUTO FLOAT BOUNDED BY (0,1)
    PARAM GAMMA = AUTO FLOAT BOUNDED BY (0,1)
    POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
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
    TUNE AUTO UNTIL(GENERATION=10)
)
UNTIL (
    GENERATION = 1000
) 
""" 

results = sw.sdl.execute(expression) 
print(results)