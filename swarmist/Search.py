from __future__ import annotations
from typing import List, Optional, Any, Callable, Tuple, Generator
from dataclasses import dataclass, field
from pymonad.either import Left, Right, Either
from Dictionary import *
from Helpers import try_catch, assert_at_least_one_nonnull, assert_equal_length
from SearchExecutor import SearchExecutor
from Exceptions import SearchEnded

def get_fit_func(fdef: FitnessFunctionDef, min_fit: Optional[float])-> Tuple[FitnessFunction, Optional[float]]:
    if fdef.minimize:
        return (fdef.func, min_fit)
    return (lambda pos: -fdef.func(pos), None if not min_fit else -min_fit)

def get_executor(space: SearchSpace, until: StopCondition)->SearchExecutor:
    fit_func, min_fit = get_fit_func(space.fit_func, until.fit)
    return SearchExecutor(
        fit_func=fit_func,
        ndims=space.ndims, 
        bounds=space.bounds,
        constraints=space.constraints,
        max_gen=until.max_gen,
        min_fit=min_fit, 
        max_evals=until.max_evals
    )

def do_init(init: Init, ctx: SearchContext)->Population:
    agents = init.method(init.population_size, ctx)
    groups = init.topology(agents) if init.topology else None 
    assert_equal_length(len(agents), "Number of created agents")
    if groups:
        pass
        #TODO check static vs dynamic topology
    groups and assert_equal_length(len(groups), "Number of neighborhoods")
    return Population(agents=agents, groups=groups, size=init.population_size)

def get_ranked_population(agents, groups)->GroupInfo:
    def rank()->GroupInfo:
        #TODO
        return None
    return Population(agents, groups, rank)

def do_update(updates: List[Update], p: Population)->Population:
    population = get_ranked_population(p.agents, p.groups)
    for update in updates:
        pass
        #TODO
        # population = 
        # method: UpdateMethod
        # selection: SelectionMethod
        # where: Conditions = None
        # limit: Limit = None

def do_search(request: SearchRequest)->SearchResults:
    executor = SearchExecutor(request.space, request.until)
    init, update = request.strategy
    res = try_catch(lambda: do_init(init, executor.context()))
    while(res.is_right()):
        res = executor.next().bind(lambda _: do_update(update, res.value))
    return SearchResults(
        results=executor.results(),
        error=res.either(lambda error: None if isinstance(error, SearchEnded) else error)
    )

def until(
        fit: Optional[float] = None, 
        max_evals: Optional[int] = None, 
        max_gen: Optional[int] = None
)->Callable[...,Either[StopCondition, Exception]]:
    def callback(): 
       _fit = float(fit) if fit else None
       _max_evals = int(max_evals) if max_evals else None
       _max_gen = int(max_gen) if max_gen else None
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
     searc_strategy: Callable[..., Either[SearchStrategy, Exception]]
)->SearchResults:
    return search_space().bind(
        lambda space: searc_strategy().bind(
            lambda strategy: stop_condition().bind(
                lambda until: Right(SearchRequest(space, strategy, until))   
            )
        )
    ).either(lambda e: SearchResults(error=e), lambda r: do_search(r))