from __future__ import annotations
from typing import List, Optional, Callable, Tuple, cast
from functools import partial, reduce
from dataclasses import replace, astuple
from pymonad.either import Right, Either
from pymonad.reader import Pipe
import numpy as np
from .dictionary import *
from .executor import SearchExecutor
from .population import get_population
from .errors import SearchEnded, try_catch, assert_at_least_one_nonnull, assert_equal_length, assert_callable

def get_fit_func(fdef: FitnessFunctionDef, min_fit: Optional[float])-> Tuple[FitnessFunction, Optional[float]]:
    if fdef.minimize:
        return (fdef.func, min_fit)
    return (lambda pos: -fdef.func(pos), None if not min_fit else -min_fit)

def get_executor(space: SearchSpace, until: StopCondition, params: Parameters)->SearchExecutor:
    fit_func, min_fit = get_fit_func(space.fit_func_def, until.fit)
    return SearchExecutor(
        fit_func=fit_func,
        ndims=space.ndims, 
        bounds=space.bounds,
        constraints=space.constraints,
        params=params,
        max_gen=until.max_gen,
        min_fit=min_fit, 
        max_evals=until.max_evals
    )

def create_agent(pos_generator: PosGenerationMethod, index: int, ctx: SearchContext)->Agent:
    pos = ctx.clip(pos_generator(ctx))
    fit = ctx.evaluate(pos)  
    return Agent(
        index=index,
        pos=pos,
        best=pos,
        delta=np.zeros(ctx.ndims),
        fit=fit,
        trials=0,
        improved=True
    )

def create_agents(method: Callable[[int]], size: int)->AgentList:
    #print(method(index=0))
    return [method(index=i) for i in range(size)]

def do_init(init: Initialization, ctx: SearchContext)->Population:
    agents = create_agents(
        partial(create_agent, ctx=ctx,pos_generator=init.generate_pos),
        init.population_size
    )
    topology = init.topology(agents) if init.topology else None
    assert_equal_length(len(agents), init.population_size, "Number of created agents")
    if topology:
        if isinstance(topology, Topology):
            assert_equal_length(len(topology), init.population_size, "Number of neighborhoods")
            topology = lambda _: topology 
        else:
            assert_callable(topology, "Dynamic topology")
    return get_population(agents)

def filter_to_update(update: Update, rank: PopulationInfo)->AgentList[Agent]:
    return update.selection(rank.info)
    # ops = Pipe(update.selection(rank.info))
    # if update.where:
    #     ops.then(lambda agents: map(update.where, agents))
    # if update.limit:
    #     ops.then(lambda agents: agents[:update.limit])
    # return ops(rank)

def do_apply(agent: Agent, executor: SearchExecutor)->Agent:
    best = agent.best
    pos = executor.clip(agent.pos)
    fit = executor.evaluate(pos)
    improved = fit < agent.fit
    best = agent.best if not improved else pos
    trials = 0 if improved else agent.trials + 1
    return replace(
        agent, 
        pos=pos,
        fit=fit,
        best=best,
        trials=trials,
        improved=improved
    )

def do_update(update: Update, agent: Agent, info: GroupInfo, executor: SearchExecutor)->Agent:    
    where = update.where 
    candidate = do_apply(
        update.method(
            UpdateContext(
                agent=agent,
                info=info,
                parameters=executor.context().parameters
            )
        ), executor)
    return candidate if not where or where(candidate) else agent
 
def do_update_all(updates: List[Update], p: Population, executor: SearchExecutor)->Either[Population, Exception]:
    def callback()->Population:
        rank = p.rank()
        new_agents = [agent for agent in p.agents]
        for update in updates: 
            to_update = filter_to_update(update, rank)
            for agent in to_update:
                new_agents[agent.index] = do_update(update, agent, rank.group_info[agent.index], executor)
        return replace(p, agents=new_agents)
    return try_catch(callback)
  
def do_search(space: SearchSpace, strategy: SearchStrategy, until: StopCondition)->SearchResults:
    executor = get_executor(space, until, strategy.parameters)
    res = try_catch(lambda: do_init(strategy.initialization, executor.context()))
    exec_update_all = partial(do_update_all, executor=executor, updates=strategy.update_pipeline)
    while(res.is_right()):
        res = executor.next().then(lambda _: exec_update_all(p=res.value))
    error = res.either(lambda e: None if isinstance(e, SearchEnded) else e, None)
    results = executor.results()
    return SearchFailed(error) if error else SearchSucceed(
        results=results,
        best=lambda: min(results, key=lambda e: e.fit),
        last=lambda: results[-1],
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

def get_request(space: SearchSpace, strategy: SearchStrategy, until: StopCondition)->SearchRequest:
    print(f"get_request={type(space.fit_func_def)}")
    return SearchRequest(strategy, space, until)

def search(
     search_space: Callable[..., Either[SearchSpace, Exception]], 
     stop_condition: Callable[..., Either[StopCondition, Exception]],
     search_strategy: Callable[..., Either[SearchStrategy, Exception]]
)->SearchResults:
    response: Either[SearchResults, Exception] = search_space().then(
        lambda space: search_strategy().then(
            lambda strategy: stop_condition().then(
                lambda until: Right(
                    do_search(
                        strategy=strategy, 
                        space=space, 
                        until=until
                    )
                )
            )
        )
    )
    return response.either(
        lambda error: SearchFailed(error=error), 
        lambda result: result
    ) 