import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw

## Differential Evolution
de_results = sw.sdl.execute("""
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X ** 2)
)
USING (
    PARAM F = 0.5
    PARAM  CR = 0.6
    POPULATION SIZE(40) INIT RANDOM_UNIFORM()
    SELECT ALL (
        USING BINOMIAL RECOMBINATION WITH PROBABILITY PARAM(CR)
        UPDATE (
            POS = PICK_RANDOM(UNIQUE) + PARAM(F) * (
                PICK_RANDOM(UNIQUE) - PICK_RANDOM(UNIQUE)
            ) 
        ) WHEN IMPROVED = TRUE
    )
)
UNTIL (
    GENERATION = 1000
) 
"""
)

print(de_results.best[-1].fit)