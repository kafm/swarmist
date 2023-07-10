import swarmist as sw
import datetime, csv
from multiprocessing import Pool
from opfunu.cec_based.cec2017 import *

numDimensions = 30
numExperiences = 30
numGen = 2500
maxEvals = numGen * 40
bounds = sw.Bounds(min=-100, max=100)
problem = F12017(ndim=numDimensions).evaluate

fips = sw.sdl.strategy("""
        PARAMETERS (
            PHI = 4.1 BOUNDED BY (0, 8)
            CHI = 0.7298 BOUNDED BY (0, 1)
        )
        POPULATION SIZE(40) INIT RANDOM_UNIFORM() WITH TOPOLOGY LBEST SIZE(2)
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
    """)
st = sw.sdl.strategy("""
        PARAMETERS (
            PHI = AUTO(BOUNDED BY (0, 8)) 
            CHI = AUTO(BOUNDED BY (0, 1))
        )
        POPULATION AUTO(BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                W = RANDOM(SIZE=2)
                PHI = SUM(W) * (PARAM(PHI) / 2)
                PM = AVG(PICK_ROULETTE(unique 2 with replacement), W)
                SCT = PHI * (PM - POS)
                POS = POS + PARAM(CHI) * (DELTA + SCT)
            ) 
        ) 
        TRY AUTO UNTIL(GENERATION=400)
    """)

st.param()
fit = sw.search(
    sw.minimize(problem, bounds, dimensions=numDimensions),
    sw.until(max_evals=maxEvals),
    sw.using(st)
)[-1].fit

fips_fit = sw.search(
    sw.minimize(problem, bounds, dimensions=numDimensions),
    sw.until(max_evals=maxEvals),
    sw.using(fips)
)[-1].fit

print(f"Fit {fit}, Fips: {fips_fit}")
