import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw

## Teaching Learning based Optimization
tlbo_results = sw.sdl.execute("""
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X ** 2)
)
USING (
    PARAM A = 2
    POPULATION SIZE(40) INIT RANDOM_UNIFORM()
    SELECT ALL (
        UPDATE (
            TF = IF_THEN(RANDOM(SIZE=1) < 0.5, 1, 2)
            POS = POS + (RANDOM() * (SWARM_BEST() - TF*AVG(ALL()))) 
        ) WHEN IMPROVED = TRUE
    )
    SELECT ALL (
        UPDATE (
            POS = REDUCE(
                PICK_RANDOM(UNIQUE),
                (ACC, REF) => ACC + IF_THEN(
                    REF.FIT < FIT,
                    RANDOM() * (REF.POS - POS),
                    RANDOM() * (POS - REF.POS)
                ), POS)
        ) WHEN IMPROVED = TRUE
    )
)
UNTIL (
    GENERATION = 1000
) 
"""
)

print(tlbo_results.best[-1].fit)