import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw

problem, bounds = sw.benchmark.sphere()
numDimensions = 20
maxGenerations = 1000

strategy_expr = """
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
"""
 
st = sw.sdl.strategy(strategy_expr) 

res = sw.search(
    sw.minimize(problem, bounds, dimensions=numDimensions),
    sw.until(max_gen=maxGenerations),
    sw.using(st),
)
print(f"res={res.best[-1].fit}")