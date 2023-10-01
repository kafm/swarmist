import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw

## Cuckoo Search
cs_results = sw.sdl.execute("""
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X ** 2)
)
USING (
    PARAM PA = 0.25
    PARAM ALPHA = 1
    PARAM MU = 1.5
    POPULATION SIZE(40) INIT RANDOM_UNIFORM()
    SELECT ALL (
        UPDATE (
            POS = POS + 
                PARAM(ALPHA) * RANDOM_LEVY(LOC=PARAM(MU)) 
                * (POS - SWARM_BEST())
        ) WHEN IMPROVED = TRUE
    )
    SELECT ALL (
        USING RECOMBINATION WITH PROBABILITY PARAM(PA)
        UPDATE (
            POS = POS + RANDOM() * (
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

print(cs_results.best[-1].fit)