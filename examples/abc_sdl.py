import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw

## Artificial Bee Colony
abc_results = sw.sdl.execute("""
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X ** 2)
)
USING (
    PARAM POP_SIZE = 40
    POPULATION SIZE(PARAM(POP_SIZE)) INIT RANDOM_UNIFORM()
    SELECT ALL (
        USING RANDOM RECOMBINATION SIZE(1)
        UPDATE (
            POS = POS + RANDOM_UNIFORM(LOW=-1, HIGH=1) * (
                POS - PICK_RANDOM()
            )
        ) WHEN IMPROVED = TRUE
    )
    SELECT SIZE(1) WHERE TRIALS > PARAM(POP_SIZE)*NDIMS 
    ORDER BY TRIALS DESC (
        INIT RANDOM_UNIFORM()
    )
)
UNTIL (
    GENERATION = 1000
) 
"""
)

print(abc_results.best[-1].fit)