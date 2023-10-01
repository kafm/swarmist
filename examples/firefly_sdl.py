import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw

## Firefly Algorithm
ff_results = sw.sdl.execute("""
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X ** 2)
)
USING (
    PARAM ALPHA = 1
    PARAM DELTA = 0.97 
    PARAM BETA = 1
    PARAM GAMMA = 0.01
    POPULATION SIZE(40) INIT RANDOM_UNIFORM()
    SELECT ALL (
        UPDATE (
            ALPHA = (PARAM(ALPHA) * PARAM(DELTA)) ** CURR_GEN
            VALUES = MAP(
                ALL(), (REF) => IF_THEN(REF.FIT < FIT, REF.POS, 0)
            )
            POS = REDUCE(
                VALUES, 
                (ACC, VAL) => ACC + (
                    (
                        PARAM(BETA) 
                        * EXP(-1 * PARAM(GAMMA) * (VAL - ACC)**2)
                    ) * (VAL - ACC) 
                    + ALPHA * RANDOM_UNIFORM(LOW=-1, HIGH=1)
                ), POS)
        )
    )
)
UNTIL (
    GENERATION = 1000
) 
"""
)

print(ff_results.best[-1].fit)