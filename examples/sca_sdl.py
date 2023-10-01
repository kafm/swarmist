import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw

## Sine cosine algorithm
sca_results = sw.sdl.execute("""
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
            SC = REPEAT(
                IF_THEN(RANDOM(SIZE=1) < 0.5, 
                    SIN( RANDOM_UNIFORM(LOW=0, 
                        HIGH=2*PI(), SIZE=1) ),
                    COS( RANDOM_UNIFORM(LOW=0, 
                        HIGH=2*PI(), SIZE=1) )
                ), NDIMS) 
            POS = POS + 
                ( A * SC * ABS( RANDOM() * SWARM_BEST() - POS ) ) 
        ) WHEN IMPROVED = TRUE
    )
)
UNTIL (
    GENERATION = 1000
) 
"""
)

print(sca_results.best[-1].fit)