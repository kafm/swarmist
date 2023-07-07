import swarmist as sw
import datetime, csv
from multiprocessing import Pool


algos = {
    "PSO": sw.sdl.strategy("""
        PARAMETERS (
            C1 = 2.05 BOUNDED BY (0, 8)
            C2 = 2.05 BOUNDED BY (0, 8) 
            CHI = 0.7298 BOUNDED BY (0, 1)
        )
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
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
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                MU= (SWARM_BEST()+BEST)/2
                SD = ABS(SWARM_BEST()-BEST)
                POS = RANDOM_NORMAL(LOC=MU, SCALE=SD)
            ) WHEN IMPROVED = TRUE
        )
    """), 
    "FIPS": sw.sdl.strategy("""
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
    """),
    "DE": sw.sdl.strategy("""
        PARAMETERS (
            F = 0.5 BOUNDED BY (0, 1)
        )
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
        SELECT ALL (
            USING BINOMIAL RECOMBINATION WITH PROBABILITY 0.6
            UPDATE (
                POS = PICK_RANDOM(UNIQUE) + PARAM(F) * (PICK_RANDOM(UNIQUE) - PICK_RANDOM(UNIQUE)) 
            ) WHEN IMPROVED = TRUE
        ) 
    """),
    "JAYA": sw.sdl.strategy("""
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
        SELECT ALL (
            UPDATE (
                ABS_POS = ABS(POS)
                POS = POS + RANDOM() * (SWARM_BEST() - ABS_POS) - RANDOM() * (SWARM_WORST() - ABS_POS)
            ) WHEN IMPROVED = TRUE
        )    
    """),
    "ABC": sw.sdl.strategy("""
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
        SELECT ALL (
            USING RANDOM RECOMBINATION SIZE(1)
            UPDATE (
                POS = RANDOM_UNIFORM(LOW=-1, HIGH=1) * (POS - PICK_RANDOM())
            ) WHEN IMPROVED = TRUE
        )
        SELECT SIZE(1) WHERE TRIALS > POPULATION_SIZE*NDIMS ORDER BY TRIALS DESC (
            INIT RANDOM_UNIFORM()
        )
    """),
    "CS": sw.sdl.strategy("""
        PARAMETERS (
            PA = 0.25 BOUNDED BY (0, 1)
            ALPHA = 1 BOUNDED BY (0, 1)
            MU = 1.5 BOUNDED BY (0,3)
        )
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
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
        PARAMETERS (
            ALPHA = 1 BOUNDED BY (0, 2)
            DELTA = 0.97 BOUNDED BY (0, 1)
            BETA = 1 BOUNDED BY (0, 2)
            GAMMA = 0.01 BOUNDED BY (0, 1)
        )
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
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
        PARAMETERS (
            A = 2 BOUNDED BY (0, 10)
        )
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
    """),
    "SCA": sw.sdl.strategy("""
        PARAMETERS (
            A = 2 BOUNDED BY (0, 10)
        )
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
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
        PARAMETERS (
            A = 2 BOUNDED BY (0, 10)
        )
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
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
        PARAMETERS (
            A = 2 BOUNDED BY (0, 10)
            BETA = .5 BOUNDED BY (0,1) 
        )
        POPULATION SIZE(40) INIT RANDOM_UNIFORM()
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
    "Sphere": sw.benchmark.sphere(),
    "Ackley": sw.benchmark.ackley(),
    "Griewank": sw.benchmark.griewank(),
    "Rastrigin": sw.benchmark.rastrigin(),
    "Schwefel": sw.benchmark.schwefel(),
    "Rosenbrock": sw.benchmark.rosenbrock(),
    "Michalewicz": sw.benchmark.michalewicz()
}
numDimensions = 30
numExperiences = 30
numGen = 1000

resultsFile = open("experiment1_results.csv", "w")
resultsWriter = csv.writer(resultsFile, lineterminator="\n")
resultsWriter.writerow(["num_exp", "problem", "algo", "fit"])


def runExperience(assets):
    num_exp, problem_name = assets
    problem_impl, problem_bounds = problems[problem_name]
    results = []
    for algo, st in algos.items():
        fit = sw.search(
                sw.minimize(problem_impl, problem_bounds, dimensions=numDimensions),
                sw.until(max_gen=numGen),
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