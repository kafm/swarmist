from __future__ import annotations
from typing import List, Optional, Type
from dataclasses import dataclass, replace
import sys
import numpy as np

@dataclass(frozen=True)
class Bounds:
    min: float
    max: float

@dataclass(frozen=True)
class StopCondition:
    min_fit: float
    max_evals: float
    max_gen: float

@dataclass(frozen=True)
class Environment:
    bounds: Bounds
    stop_condition: StopCondition
    fitness_function: callable

@dataclass
class Environment:
    bounds: Bounds
    stop_condition: StopCondition
    fitness_function: callable
    ensure_constraints: callable


@dataclass(frozen=True)
class Individual:
    index: int
    pos: float
    lbest: float
    fit: float
    neighborhood: Neighboarhood

class Neighboarhood(list):
    def __init__(self, item: Optional[Individual] = None, *args: Type[int], **kwargs: Type[int]):
        if item is not None:
            if not isinstance(item, Individual):
                raise TypeError("Item must be an individual")
            self.append(item)
        super().__init__(*args, **kwargs)

    def map(self, callback: callable):
        pass

    def filter(self, callback: callable):
        pass

    #TODO...

class Population(Neighboarhood):
   def __init__(self, *args: Type[int], **kwargs: Type[int]):
        super().__init__(*args, **kwargs)


def init_population(size: int, callback: callable)->Population:
    def init_pop_helper(size: int, population: Population):
        if size == 0: 
            return population
        population.insert(0, callback(size))
        return init_pop_helper(size - 1, population)
    return init_pop_helper(size, Population())

def gbest_topology(population: Population)->Population:
    return population.map(lambda individual: replace(individual, neighborhood=population))



### THE ALGO ###
environment = Environment(
    Bounds(0,100),
    fitness_function=lambda pos: pos**2,
    ensure_constraints=lambda pos: np.clip(pos, 0, 100) #TODO find a cleanner way

)

class SearchStrategy:
    population_size: int
    topology: callable
    init_method: callable
    search: callable

class SearchSpace:
    bounds: callable
    fitness_function: callable
    constraints: callable

class SearchRequest: 
    strategy: SearchStrategy
    space: SearchSpace
    stop_condition: StopCondition
  
class SearchResults: 
    pass

def update_method1(individual: Individual, fit_func: callable)->Individual:
    pass

def update_method2(individual: Individual, fit_func: callable)->Individual:
    pass

def search(population: Population, fit_func: callable)->Population:
    condition = lambda : True
    population = population.map(update_method1) | population.partial_map(update_method2, condition) # TODO see how can I make it a reality

def search_until(request: SearchRequest, stop_condition: callable)->SearchResults:
    def search_until_helper(fit: float, gen:int, evals: int)->SearchResults:
        pass
    return search_until_helper(sys.float_info.max, 1, 0, 0)
    

def algo(search_strategy: SearchStrategy, search_space: SearchSpace)->SearchResults:
    population = search_strategy.topology(
        init_population(search_strategy.population_size,search_strategy.init_method) #TODO take data from env and initialize Individual in init population. 
    )
    def evaluate():
        pass 
    return search(search_strategy, evaluate)



environment.repeatUntil(StopCondition(max_gen=100), lambda : search(population))



bounds = Bounds(0,1)
bounds._rereplace()

StopCondition = namedtuple('StopCondition', ['min_fit', 'max_evals', 'max_gen'])
Context = namedtuple('Context', ['bounds', 'stop_condition', 'fit_func'])
Individual = namedtuple('Individual', ['pos', 'p_best', 'fitness'])
pop_size = 50
population = [Individual(1) for _ in pop_size]

def update_method():
    pass
map(population, update_method)

# class Individual:
#     def __init__(
#       self
#     ):
#         pass

# class Population: 
#     def __init__(
#       self,
#       size, 
#       initMethod
#     ):
#         pass
  
# class StopCondition: 
#     def __init__(
#         self,
#         max_evalution: int, #Either monad
#         min_fitness: float,  #Either monad
#         max_generations: int,  #Either monad
#     ):
#         pass

# class Environment: 
#     def __init__(
#         self,
#         stop_condition: StopCondition,
#         fitness_function: callable
#     ):
#         pass

# def initMethod(index): #can be any function
#     pass

# def Topology(pop): #can be any function that receives a list of individuals
#     pass

# def Search(dim, fit_func):
#     pass

# def fitFunc():
#     pass

# def update_method():
#     recombination(updateMethod) * centroid() * referencePoint()


def create_individual():
    pass

def gbest():
    pass

def stop_condition(ctx): #ctx.min_fit, ctx.max_eval, ctx.max_gen
    pass
    #return ctx.min_fit <= 1

def reference_point(ind): 
    pass

def centroid(ind):
    pass

def search_region(ind):
    pass

def perturbation(ind):
    pass

def recombination(ind):
    pass

def repeat_until():
    def func():
        pass
    return func 

def update_method_builder(fit_func):
    def evaluate(individual):
        fit = fit_func(individual.pos)
        #TODO
        pass
    return lambda individual : individual | reference_point | centroid | search_region | perturbation | recombination | evaluate

def search_method(population, update_method):
    map(population, update_method)
    map(filter(population, lambda i: True), update_method)
    #...can be several stages

def evolve(fit_func, ndims, pop_size):
    population = gbest([create_individual(i, ndims) for i in pop_size])
    update_method = update_method_builder(fit_func)
    repeat_until(stop_condition)(lambda: search_method(population, update_method))
    return population.best()



class Indi
