import mealpy as mp
import numpy as np
import datetime, csv
from multiprocessing import Pool
from opfunu.cec_based.cec2017 import *

numDimensions = 30
numExperiences = 30
numGen = 2500
numEvals = 100000
populationSize = 40

algos = {
    "JAYA": mp.JA.OriginalJA(epoch=numGen, pop_size=populationSize)
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

resultsFile = open("experiment3_mealpy_results_jaya_fix.csv", "w")
resultsWriter = csv.writer(resultsFile, lineterminator="\n")
resultsWriter.writerow(["num_exp", "problem", "algo", "fit"])


def runExperience(assets):
    num_exp, problem_name = assets
    results = []
    for algo, model in algos.items():
        _, best_fitness = model.solve({
            "fit_func": problems[problem_name],
            "lb": [-100, ] * numDimensions,
            "ub": [100, ] * numDimensions,
            "minmax": "min",
            "log_to": None,
            "save_population": False,
        })
        results.append([num_exp, problem_name, algo, best_fitness])
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


