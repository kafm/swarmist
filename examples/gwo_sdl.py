import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw

## Grey wolf optimizer
gwo_results = sw.sdl.execute("""
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X ** 2)
)
USING (
    PARAM A = 2
    POPULATION SIZE(40) INIT RANDOM_UNIFORM()
    SELECT ALL (
        UPDATE (
            A = PARAM(A) - CURR_GEN * ( PARAM(A) / MAX_GEN ) 
            POS = AVG(
                MAP(
                    SWARM_BEST(3), (REF) => 
                        A * ABS( (2 * RANDOM()) * REF.POS - POS)
                )
            )
        ) WHEN IMPROVED = TRUE
    )
)
UNTIL (
    GENERATION = 1000
) 
"""
)

print(gwo_results.best[-1].fit)