import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw

## Differential Evolution auto tunning in sdl environment
res = sw.sdl.execute("""
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X ** 2)
)
USING (
    PARAM F = AUTO FLOAT BOUNDED BY (0, 1)
    PARAM  CR = AUTO FLOAT BOUNDED BY (0, 1)
    POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
    SELECT ALL (
        USING BINOMIAL RECOMBINATION WITH PROBABILITY PARAM(CR)
        UPDATE (
            POS = PICK_RANDOM(UNIQUE) + PARAM(F) * (
                PICK_RANDOM(UNIQUE) - PICK_RANDOM(UNIQUE)
            ) 
        ) WHEN IMPROVED = TRUE
    )
    TUNE AUTO UNTIL(GENERATION=50)
)
UNTIL (
    GENERATION = 1000
) 
"""
)

print(f"fit={res.fit}, params={res.parameters}")