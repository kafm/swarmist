from __future__ import annotations
from typing import List, Optional, Callable, Tuple, cast
from functools import partial, reduce
from dataclasses import replace, astuple
from pymonad.either import Right, Either
from pymonad.reader import Pipe
import numpy as np
from swarmist.core.dictionary import *
from swarmist.core.executor import SearchExecutor
from swarmist.core.evaluator import Evaluator
from swarmist.core.population import Population
from swarmist.core.errors import try_catch, assert_at_least_one_nonnull
from swarmist.strategy import Strategy
from swarmist.space import SpaceBuilder

def init_population(strategy: SearchStrategy, ctx: SearchContext)->Either[Exception, Population]:
    return try_catch(
        lambda: Population(
            strategy=strategy,
            ctx=ctx
        )
    )

def do_search(                   
    strategy: SearchStrategy, 
    evaluator: Evaluator,
    until: StopCondition
)->Either[Exception, SearchResults]:
    executor = SearchExecutor( 
        evaluator=evaluator,
        parameters=strategy.parameters,
        max_gen=until.max_gen,
        min_fit=until.fit,
        max_evals=until.max_evals
    )  
    return init_population(
        strategy, executor.context()
    ).then(executor.evolve)

def until(
        fit: Optional[float] = None, 
        max_evals: Optional[int] = None, 
        max_gen: Optional[int] = None
)->Callable[...,Either[Exception, StopCondition]]:
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

def using(strategy: Strategy)->Either[Exception, Callable[..., SearchStrategy]]:
    return strategy.get

def search(
     space: SpaceBuilder, 
     until: Callable[...,Either[Exception, StopCondition]],
     strategy: Callable[..., Either[Exception, SearchStrategy]],
)->SearchResults:
    def raise_error(e: Exception):
        raise e
    res: Either[Exception, SearchResults] = strategy().then(
        lambda search_strategy: until().then(
            lambda stop_condition: do_search(
                strategy=search_strategy,
                evaluator=space.get(),
                until=stop_condition
            )
        )
    )
    return res.either(raise_error, lambda res: res)