from __future__ import annotations
from typing import List, Optional, Callable, Tuple, cast
from functools import partial, reduce
from dataclasses import replace, astuple
from pymonad.either import Right, Either
from pymonad.reader import Pipe
import numpy as np
from .dictionary import *
from .executor import SearchExecutor
from .population import update_agent, init_population, get_population_rank
from .errors import SearchEnded, try_catch, assert_at_least_one_nonnull

def get_fit_func(fdef: FitnessFunctionDef, min_fit: Optional[float])-> Tuple[FitnessFunction, Optional[float]]:
    if fdef.minimize:
        return (fdef.func, min_fit)
    return (lambda pos: -fdef.func(pos), None if not min_fit else -min_fit)

def get_executor(space: SearchSpace, until: StopCondition)->SearchExecutor:
    fit_func, min_fit = get_fit_func(space.fit_func_def, until.fit)
    return SearchExecutor(
        fit_func=fit_func,
        ndims=space.ndims, 
        bounds=space.bounds,
        constraints=space.constraints,
        max_gen=until.max_gen,
        min_fit=min_fit, 
        max_evals=until.max_evals
    )

def filter_to_update(update: Update, rank: PopulationInfo)->AgentList[Agent]:
    return update.selection(rank.info)
    # ops = Pipe(update.selection(rank.info))
    # if update.where:
    #     ops.then(lambda agents: map(update.where, agents))
    # if update.limit:
    #     ops.then(lambda agents: agents[:update.limit])
    # return ops(rank)

def get_pipeline_executor(updates: List[Update], executor: SearchExecutor)->Callable[[Population], Population]:
    n = len(updates)
    def callback(p: Population, index: int = 0)->Population:
        if index >= n: 
            return p
        update: Update = updates[index]
        rank = get_population_rank(p)
        new_agents = p.agents.copy()
        to_update = filter_to_update(update, rank) 
        for agent in to_update:
            new_agents[agent.index] = update_agent(
                update, 
                agent, 
                rank.group_info[agent.index], 
                executor
            )
        return callback(replace(p, agents=new_agents), index+1)
    return callback

def get_search_method(
    strategy: SearchStrategy, 
    executor: SearchExecutor, 
    population: Population
)->Callable[[Population], Either[SearchResults, Exception]]:
    pipeline = get_pipeline_executor(strategy.update_pipeline, executor)
    new_pop: Population = population
    def callback():
        nonlocal new_pop
        new_pop = pipeline(new_pop)
    return executor.run(callback)

def do_search(                   
    strategy: SearchStrategy, 
    space: SearchSpace,
    until: StopCondition
)->Either[Population, Exception]:
    executor = get_executor(space, until)
    return init_population(strategy.initialization, executor.context()).then(
        lambda population: get_search_method(
            strategy=strategy, 
            executor=executor,
            population=population
        )
    )

def until(
        fit: Optional[float] = None, 
        max_evals: Optional[int] = None, 
        max_gen: Optional[int] = None
)->Callable[...,Either[StopCondition, Exception]]:
    def callback(): 
       _fit = None if not fit else float(fit)
       _max_evals = None if not max_evals else int(max_evals)
       _max_gen = None if  not max_gen else int(max_gen)
       assert_at_least_one_nonnull({
           "fitness max/min": _fit, 
           "maximum number of evaluations": _max_evals,
           "maximum number of generations": _max_gen
       })
       return StopCondition(_fit, _max_evals, _max_gen)
    return lambda: try_catch(callback)

def search(
     search_space: Callable[..., Either[SearchSpace, Exception]], 
     stop_condition: Callable[..., Either[StopCondition, Exception]],
     search_strategy: Callable[..., Either[SearchStrategy, Exception]]
)->Either[SearchResults, Exception]:
    return search_space().then(
        lambda space: search_strategy().then(
            lambda strategy: stop_condition().then(
                lambda until: do_search(
                        strategy=strategy, 
                        space=space,
                        until=until
                    )
            )
        )
    )
