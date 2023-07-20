import swarmist as sw
import datetime, csv
from multiprocessing import Pool
from opfunu.cec_based.cec2017 import *

numDimensions = 30
numExperiences = 30
numEvals = 100000
bounds = sw.Bounds(min=-100, max=100)

algos = {
    "SCA": sw.sdl.strategy("""
        PARAM A = 2
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                A = PARAM(A) - CURR_GEN * ( PARAM(A) / MAX_GEN )
                SC = REPEAT(
                    IF_THEN(
                        RANDOM(SIZE=1) < 0.5, 
                        SIN( 2 * PI() * RANDOM(SIZE=1) ),
                        COS( 2 * PI() * RANDOM(SIZE=1) )
                    ), 
                    NDIMS) 
                POS = POS + ( A * SC * ABS( 2 * RANDOM() * SWARM_BEST() - POS ) ) 
            ) WHEN IMPROVED = TRUE
        )    
    """),
    "GWO": sw.sdl.strategy("""
        PARAM  A = 2
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                A = PARAM(A) - 2 * CURR_GEN / MAX_GEN
                POS = AVG(
                        MAP(
                            SWARM_BEST(3), (REF) => REF.POS
                                - (A * (2 * RANDOM(SIZE=NDIMS) - 1))
                                * ABS((2 * RANDOM(SIZE=NDIMS)) * REF.POS - POS)
                        )
                    )
            ) WHEN IMPROVED = TRUE
        ) 
    """),
    "WO": sw.sdl.strategy("""
        PARAM A = 2
        PARAM BETA = 1
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                r = RANDOM(SIZE=1)
                a = PARAM(A) - 2 * CURR_GEN / MAX_GEN
                l =  RANDOM_UNIFORM(LOW=-1, HIGH=1, SIZE=1)
                A = 2 * a * r - a
                C = 2 * r
                POS = IF_THEN(
                    RANDOM(SIZE=1) < .5,
                    APPLY(
                        IF_THEN(ABS(A) < 1, SWARM_BEST(), RANDOM_POS()),
                        (REF) => REF - A * ABS(C * REF - POS)
                    ),
                    SWARM_BEST() + EXP(PARAM(BETA) * l) * COS(2 * PI() * l) * ABS(SWARM_BEST() - POS)
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

resultsFile = open("experiment3_results_fix.csv", "w")
resultsWriter = csv.writer(resultsFile, lineterminator="\n")
resultsWriter.writerow(["num_exp", "problem", "algo", "fit"])


def runExperience(assets):
    num_exp, problem_name = assets
    results = []
    for algo, st in algos.items():
        fit = sw.search(
                sw.minimize(problems[problem_name], bounds, dimensions=numDimensions),
                sw.until(max_evals=numEvals),
                sw.using(st)
            )[-1].fit
        results.append([num_exp, problem_name, algo, fit])
    return results

if __name__ == '__main__':
    with Pool() as pool:
        for i in range(numExperiences):
            num_exp = [i+1]
            print("[{time}] Starting experience no {no}".format(no=num_exp, time=datetime.datetime.now()))
            for res in pool.map(runExperience, [(num_exp, key) for key in problems.keys()]):
                resultsWriter.writerows(res)
            print("[{time}] Ended experience no {no}".format(no=num_exp, time=datetime.datetime.now()))
                
resultsFile.close()
