import swarmist as sw
import datetime, csv
from multiprocessing import Pool
from opfunu.cec_based.cec2017 import *

numDimensions = 30
numExperiences = 30
numEvals = 100000
bounds = sw.Bounds(min=-100, max=100)
tuneUntil = 1 #100

algos = {
    "PSO": sw.sdl.strategy("""
        PARAM C1 = AUTO FLOAT BOUNDED BY (0, 8)
        PARAM C2 = AUTO FLOAT BOUNDED BY (0, 8) 
        PARAM CHI = AUTO FLOAT BOUNDED BY (0, 1)
        POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                VELOCITY= PARAM(CHI) * (
                    DELTA 
                    + PARAM(C1) * RANDOM() * (BEST-POS)
                    + PARAM(C2) * RANDOM() * (SWARM_BEST()-POS)
                )
                POS = POS + VELOCITY
            ) 
        )
    """), 
    "BB": sw.sdl.strategy("""
        POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                MU= (SWARM_BEST()+BEST)/2
                SD = ABS(SWARM_BEST()-BEST)
                POS = RANDOM_NORMAL(LOC=MU, SCALE=SD)
            ) WHEN IMPROVED = TRUE
        )
    """), 
    "FIPS": sw.sdl.strategy("""
        PARAM PHI = AUTO FLOAT BOUNDED BY (0, 8)
        PARAM CHI = AUTO FLOAT BOUNDED BY (0, 1)
        POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM() WITH TOPOLOGY LBEST SIZE(2)
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
    """),
    "DE": sw.sdl.strategy("""
        PARAM F = AUTO FLOAT BOUNDED BY (0, 1)
        PARAM CR = AUTO FLOAT BOUNDED BY (0, 1)
        POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
        SELECT ALL (
            USING BINOMIAL RECOMBINATION WITH PROBABILITY PARAM(CR)
            UPDATE (
                POS = PICK_RANDOM(UNIQUE) + PARAM(F) * (PICK_RANDOM(UNIQUE) - PICK_RANDOM(UNIQUE)) 
            ) WHEN IMPROVED = TRUE
        )
    """),
    "JAYA": sw.sdl.strategy("""
        POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                ABS_POS = ABS(POS)
                POS = POS + RANDOM() * (SWARM_BEST() - ABS_POS) - RANDOM() * (SWARM_WORST() - ABS_POS)
            ) WHEN IMPROVED = TRUE
        )
    """),
    "ABC": sw.sdl.strategy("""
        PARAM POP_SIZE = AUTO INT BOUNDED BY (10, 100)
        PARAM NUM_TRIALS = AUTO INT BOUNDED BY (10, 300)
        POPULATION SIZE(PARAM(POP_SIZE)) INIT RANDOM_UNIFORM()
        SELECT ALL (
            USING RANDOM RECOMBINATION SIZE(1)
            UPDATE (
                POS = RANDOM_UNIFORM(LOW=-1, HIGH=1) * (POS - PICK_RANDOM())
            ) WHEN IMPROVED = TRUE
        )
        SELECT SIZE(1) WHERE TRIALS > PARAM(NUM_TRIALS) ORDER BY TRIALS DESC (
            INIT RANDOM_UNIFORM()
        )
    """),
    "CS": sw.sdl.strategy("""
        PARAM PA = AUTO FLOAT BOUNDED BY (0, 1)
        PARAM ALPHA = AUTO FLOAT BOUNDED BY (0, 1)
        PARAM MU = AUTO FLOAT BOUNDED BY (0,3)
        POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                POS = POS + PARAM(ALPHA) * RANDOM_LEVY(LOC=PARAM(MU)) * (POS - SWARM_BEST())
            ) WHEN IMPROVED = TRUE
        )
        SELECT ALL (
            USING RECOMBINATION WITH PROBABILITY PARAM(PA)
            UPDATE (
                POS = POS + RANDOM() * (PICK_RANDOM(UNIQUE) - PICK_RANDOM(UNIQUE))
            ) WHEN IMPROVED = TRUE
        )
    """),
    "FF": sw.sdl.strategy("""
        PARAM ALPHA = AUTO FLOAT BOUNDED BY (0, 2)
        PARAM DELTA = AUTO FLOAT BOUNDED BY (0, 1)
        PARAM BETA = AUTO FLOAT BOUNDED BY (0, 2)
        PARAM GAMMA = AUTO FLOAT BOUNDED BY (0, 1)
        POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                ALPHA = ( PARAM(ALPHA) * PARAM(DELTA) ) ** CURR_GEN
                VALUES = MAP(ALL(), (REF) => IF_THEN(REF.FIT < FIT, REF.POS, 0))
                POS = REDUCE(
                    VALUES, 
                    (ACC, VAL) => ACC + (
                        ( PARAM(BETA) * EXP( -1 * PARAM(GAMMA) * (VAL - ACC)**2 ))
                        * (VAL - ACC) + ALPHA * RANDOM_UNIFORM(LOW=-1, HIGH=1)
                    ),
                    POS
                )
            )
        )
    """),
    "GWO": sw.sdl.strategy("""
        PARAM A = AUTO FLOAT BOUNDED BY (0, 10)
        POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
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
    """),
    "SCA": sw.sdl.strategy("""
        PARAM A = AUTO FLOAT BOUNDED BY (0, 10)
        POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                A = PARAM(A) - CURR_GEN * ( PARAM(A) / MAX_GEN )
                SC = REPEAT(
                    IF_THEN(
                        RANDOM(SIZE=1) < 0.5, 
                        SIN( RANDOM_UNIFORM(LOW=0, HIGH=2*PI(), SIZE=1) ),
                        COS( RANDOM_UNIFORM(LOW=0, HIGH=2*PI(), SIZE=1) )
                    ), 
                    NDIMS) 
                POS = POS + ( A * SC * ABS( RANDOM() * SWARM_BEST() - POS ) ) 
            ) WHEN IMPROVED = TRUE
        )
    """),
    "TLBO": sw.sdl.strategy("""
        PARAM A = AUTO FLOAT BOUNDED BY (0, 10)
        POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                TF = IF_THEN(RANDOM(SIZE=1) < 0.5, 1, 2)
                POS = POS + ( RANDOM() * ( SWARM_BEST() - TF*AVG(ALL()) ) ) 
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
                    ),
                    POS
                )
            ) WHEN IMPROVED = TRUE
        )
    """),
    "WO": sw.sdl.strategy("""
        PARAM A = AUTO FLOAT BOUNDED BY (0, 10)
        PARAM BETA = AUTO FLOAT BOUNDED BY (0,1) 
        POPULATION SIZE(AUTO INT BOUNDED BY (10, 100)) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                a = PARAM(A) - CURR_GEN * ( PARAM(A) / MAX_GEN )
                A = 2 * ( a * RANDOM() ) - a
                ATTACK = IF_THEN(RANDOM(SIZE=1) <.5, TRUE, FALSE)
                REF = IF_THEN( ATTACK = TRUE AND NORM(A) < 1, PICK_RANDOM(UNIQUE), SWARM_BEST() )
                L = IF_THEN(ATTACK = TRUE,  RANDOM_UNIFORM(LOW=-1, HIGH=1), 0)
                D = IF_THEN(
                    ATTACK = TRUE, 
                    ABS( REF - POS ), 
                    ABS( (2 * RANDOM()) * REF - POS )
                )
                POS = IF_THEN(
                    ATTACK = TRUE,
                    D * EXP( PARAM(BETA) * L ) * COS( 2 * PI() * L ) + REF,
                    REF - A * D
                )
            ) WHEN IMPROVED = TRUE
        )
    """)
}

problems = {
    "F1": F12017(ndim=numDimensions).evaluate,
    "F2": F22017(ndim=numDimensions).evaluate,
    "F3": F32017(ndim=numDimensions).evaluate,
    "F4": F42017(ndim=numDimensions).evaluate,
    "F5": F52017(ndim=numDimensions).evaluate,
    "F6": F62017(ndim=numDimensions).evaluate,
    "F7": F72017(ndim=numDimensions).evaluate,
    "F8": F82017(ndim=numDimensions).evaluate,
    "F9": F92017(ndim=numDimensions).evaluate,
    "F10": F102017(ndim=numDimensions).evaluate,
    "F11": F112017(ndim=numDimensions).evaluate,
    "F12": F122017(ndim=numDimensions).evaluate,
    "F13": F132017(ndim=numDimensions).evaluate,
    "F14": F142017(ndim=numDimensions).evaluate,
    "F15": F152017(ndim=numDimensions).evaluate,
    "F16": F162017(ndim=numDimensions).evaluate,
    "F17": F172017(ndim=numDimensions).evaluate,
    "F18": F182017(ndim=numDimensions).evaluate,
    "F19": F192017(ndim=numDimensions).evaluate,
    "F20": F202017(ndim=numDimensions).evaluate
}

resultsFile = open("experiment4_results.csv", "w")
resultsWriter = csv.writer(resultsFile, lineterminator="\n")
resultsWriter.writerow(["num_exp", "problem", "algo", "fit"])


def runExperience(num_exp, problem_name):
    results = []
    for algo, st in algos.items():
        print(f"algo now is {algo}")
        fit = sw.search(
                sw.minimize(problems[problem_name], bounds, dimensions=numDimensions),
                sw.until(max_evals=numEvals),
                sw.tune(st, max_gen=tuneUntil)
            ).fit
        results.append([num_exp, problem_name, algo, fit])
    return results

for i in range(numExperiences):
    num_exp = [i+1]
    print("[{time}] Starting experience no {no}".format(no=num_exp, time=datetime.datetime.now()))
    for key in problems.keys():
        res = runExperience(num_exp, key)
        resultsWriter.writerows(res)
    print("[{time}] Ended experience no {no}".format(no=num_exp, time=datetime.datetime.now()))
       
                
resultsFile.close()
