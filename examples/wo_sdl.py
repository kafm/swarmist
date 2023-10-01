import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw

# Whale Optimization
wo_results = sw.sdl.execute("""
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X ** 2)
)
USING (
    PARAM A = 2
    PARAM BETA = 0.5
    POPULATION SIZE(40) INIT RANDOM_UNIFORM()
    SELECT ALL (
        UPDATE (
            a = PARAM(A) - CURR_GEN * (PARAM(A) / MAX_GEN)
            A = 2 * ( a * RANDOM() ) - a
            ATTACK = IF_THEN(RANDOM(SIZE=1) <.5, TRUE, FALSE)
            REF = IF_THEN( ATTACK = TRUE AND NORM(A) < 1, 
                PICK_RANDOM(UNIQUE), SWARM_BEST() )
            L = IF_THEN(ATTACK = TRUE, 
                RANDOM_UNIFORM(LOW=-1, HIGH=1), 0)
            D = IF_THEN(ATTACK = TRUE, 
                ABS( REF - POS ), ABS( (2 * RANDOM()) * REF - POS )
            )
            POS = IF_THEN(
                ATTACK = TRUE,
                D * EXP( PARAM(BETA) * L ) 
                * COS( 2 * PI() * L ) + REF,
                REF - A * D
            )
        ) WHEN IMPROVED = TRUE
    )
)
UNTIL (
    GENERATION = 1000
) 
"""
)

print(wo_results.best[-1].fit)