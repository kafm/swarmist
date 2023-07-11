import mealpy as mp
import numpy as np
import datetime
from opfunu.cec_based.cec2017 import *

numDimensions = 30
numExperiences = 30
numGen = 2500
numEvals = 100000
populationSize = 40

def fitness_function(solution):
    return np.sum(solution**2)

problem = {
    "fit_func": fitness_function,
    "lb": [-100, ] * 30,
    "ub": [100, ] * 30,
    "minmax": "min",
    "log_to": None,
    "save_population": False,
}

## Run the algorithm
model = mp.SMA.BaseSMA(epoch=100, pop_size=50, pr=0.03)
best_position, best_fitness = model.solve(problem)
print(f"Best solution: {best_position}, Best fitness: {best_fitness}")


