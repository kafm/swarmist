from __future__ import annotations
from typing import Optional, Callable
from functools import partial
from pymonad.either import Right, Either
from swarmist.core.dictionary import *
from swarmist.core.executor import SearchExecutor
from swarmist.core.evaluator import Evaluator
from swarmist.core.population import Population
from swarmist.core.errors import try_catch, assert_at_least_one_nonnull
from swarmist.strategy import Strategy
from swarmist.space import SpaceBuilder
from swarmist.tune import Tune
import multiprocessing


def init_population(
    strategy: SearchStrategy, ctx: SearchContext
) -> Either[Exception, Population]:
    return try_catch(lambda: Population(strategy=strategy, ctx=ctx))


def do_search(
    strategy: SearchStrategy, evaluator: Evaluator, until: StopCondition
) -> Either[Exception, SearchResults]:
    executor = SearchExecutor(
        evaluator=evaluator,
        parameters=strategy.parameters,
        max_gen=until.max_gen,
        min_fit=until.fit,
        max_evals=until.max_evals,
        population_size=strategy.initialization.population_size,
    )
    return init_population(strategy, executor.context()).then(executor.evolve)


def do_tune(
    strategy: TuneStrategy, evaluator: Evaluator, until: StopCondition
) -> Either[Exception, TuneResults]:
    # n_jobs = (
    #     None if strategy.num_jobs == None else strategy.num_jobs
    # )
    # print(f"Tunning with {n_jobs} jobs")
    if strategy.num_jobs != None:
        print(f"Tunning with {strategy.num_jobs} jobs")
    return Tune(
        strategy, partial(do_search, evaluator=evaluator, until=until)
    ).optimize(jobs=strategy.num_jobs)


def do_tune_or_search(
    strategy: Union[SearchStrategy, TuneStrategy],
    evaluator: Evaluator,
    until: StopCondition,
) -> Either[Exception, Union[SearchResults, TuneResults]]:
    return (
        do_tune(strategy, evaluator, until)
        if isinstance(strategy, TuneStrategy)
        else do_search(strategy, evaluator, until)
    )


def until(
    fit: Optional[float] = None,
    max_evals: Optional[int] = None,
    max_gen: Optional[int] = None,
) -> Callable[..., Either[Exception, StopCondition]]:
    def callback():
        _fit = None if not fit else float(fit)
        _max_evals = None if not max_evals else int(max_evals)
        _max_gen = None if not max_gen else int(max_gen)
        assert_at_least_one_nonnull(
            {
                "fitness max/min": _fit,
                "maximum number of evaluations": _max_evals,
                "maximum number of generations": _max_gen,
            }
        )
        return StopCondition(_fit, _max_evals, _max_gen)

    return lambda: try_catch(callback)


def using(strategy: Strategy) -> Either[Exception, Callable[..., SearchStrategy]]:
    return strategy.get


def tune(
    strategy: Strategy, max_gen: int = 50, num_jobs: Optional[int] = None
) -> Either[Exception, Callable[..., TuneStrategy]]:
    return lambda: strategy.get().then(
        lambda st: Right(TuneStrategy(strategy=st, max_gen=max_gen, num_jobs=num_jobs))
    )


def search(
    space: SpaceBuilder,
    until: Callable[..., Either[Exception, StopCondition]],
    strategy: Callable[..., Either[Exception, SearchStrategy]],
) -> Union[SearchResults, TuneResults]:
    def raise_error(e: Exception):
        raise e

    res: Either[Exception, SearchResults] = strategy().then(
        lambda search_strategy: until().then(
            lambda stop_condition: do_tune_or_search(
                strategy=search_strategy, evaluator=space.get(), until=stop_condition
            )
        )
    )
    return res.either(raise_error, lambda res: res)
