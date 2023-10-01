import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw

# Canonical PSO
pso_results = sw.sdl.execute("""
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X ** 2)
)
USING (
    PARAM C1 = 2.05
    PARAM C2 = 2.05
    PARAM CHI = 0.7298
    POPULATION SIZE(40) INIT RANDOM_UNIFORM()
    SELECT ALL (
        UPDATE (
            VELOCITY= PARAM(CHI) * (DELTA 
                + PARAM(C1) * RANDOM() * (BEST-POS)
                + PARAM(C2) * RANDOM() * (SWARM_BEST()-POS))
            POS = POS + VELOCITY
        ) 
    )
)
UNTIL (
    GENERATION = 1000
) 
"""
)

# Barebones PSO
bb_results = sw.sdl.execute("""
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X ** 2)
)
USING (
    POPULATION SIZE(40) INIT RANDOM_UNIFORM()
    SELECT ALL (
        UPDATE (
            MU= (SWARM_BEST()+BEST)/2
            SD = ABS(SWARM_BEST()-BEST)
            POS = RANDOM_NORMAL(LOC=MU, SCALE=SD)
        ) WHEN IMPROVED = TRUE
    )
)
UNTIL (
    GENERATION = 1000
) 
"""
)

# Fully Informed Particle Swarm
fips_results = sw.sdl.execute("""
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X ** 2)
)
USING (
    PARAM PHI = 4.1
    PARAM CHI = 0.7298
    POPULATION SIZE(40) INIT RANDOM_UNIFORM() 
        WITH TOPOLOGY LBEST SIZE(2)
    SELECT ALL (
        UPDATE (
            NEIGHBORS = NEIGHBORHOOD()
            N = COUNT(NEIGHBORS)
            W = RANDOM(SIZE=N)
            PHI = SUM(W) * (PARAM(PHI) / N)
            PM = AVG(NEIGHBORS, W)
            SCT = PHI * (PM - POS)
            POS = POS + PARAM(CHI) * (DELTA + SCT)
        ) 
    )
)
UNTIL (
    GENERATION = 1000
) 
"""
)

print(fips_results.best[-1].fit, bb_results.best[-1].fit, pso_results.best[-1].fit)