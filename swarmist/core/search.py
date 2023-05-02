from __future__ import annotations
from typing import List, Optional, Callable, Tuple
from functools import partial
from dataclasses import replace
from pymonad.either import Right, Either
from pymonad.reader import Pipe
from .dictionary import *
from .executor import SearchExecutor
from .population import get_population
from .errors import SearchEnded, try_catch, assert_at_least_one_nonnull, assert_equal_length, assert_function_signature

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

def do_init(init: Init, ctx: SearchContext)->Population:
    agents = init.method(init.population_size, ctx)
    topology = init.topology(agents) if init.topology else None
    assert_equal_length(len(agents), init.population_size, "Number of created agents")
    if topology:
        if isinstance(topology, Topology):
            assert_equal_length(len(topology), init.population_size, "Number of neighborhoods")
            topology = lambda _: topology 
        elif callable(topology): 
            assert_function_signature(topology, DynamicTopology, "Dynamic topology")   
        else: 
            raise Exception("Wrong instance of topology provided. Expecting either static or dynamic topology") 
    return get_population(agents)

def filter_to_update(update: Update, rank: PopulationInfo)->AgentList[Agent]:
    ops = Pipe(update.selection(rank.info))
    if update.where:
        ops.then(lambda agents: map(update.where, agents))
    if update.limit:
        ops.then(lambda agents: agents[:update.limit])
    return ops(rank)

def do_apply(agent: Agent, executor: SearchExecutor)->Agent:
    best = agent.best
    pos = executor.clip(agent.pos)
    fit = executor.evaluate(pos)
    improved = fit < agent.fit
    best = agent.best if not improved else pos
    candidate = Agent(
        delta=agent.delta,
        pos=pos,
        fit=fit,
        bfit=min(fit, agent.fit),
        best=best,
        improved=improved
    )
    return candidate

def do_update(update: Update, agent: Agent, info: GroupInfo, executor: SearchExecutor)->Agent:    
    where = update.where 
    candidate = do_apply(
        update.method(
            UpdateContext(
                agent=agent,
                info=info,
                parameters=executor.ctx.parameters
            )
        ), executor)
    return candidate if not where or where(candidate) else agent
 
def do_update_all(updates: List[Update], p: Population, executor: SearchExecutor)->Population:
    rank = p.rank()
    new_agents = [agent for agent in p.agents]
    for update in updates: 
        to_update = filter_to_update(update, rank)
        for i in range(p.size):
            agent = new_agents[i]
            new_agents[i] = do_update(update, agent, rank.group_info[i], executor) if agent in to_update else agent
    return replace(p, agents=new_agents)
  
def do_search(request: SearchRequest)->SearchResults:
    init, update, params = request.strategy
    executor = get_executor(request.space, request.until, params)
    res = try_catch(lambda: do_init(init, executor.context()))
    exec_update_all = partial(do_update_all, executor=executor)
    while(res.is_right()):
        res = executor.next().bind(lambda _: exec_update_all(update, res.value))
    error = res.either(lambda e: None if isinstance(e, SearchEnded) else e)
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

def search(
     search_space: Callable[..., Either[SearchSpace, Exception]], 
     stop_condition: Callable[..., Either[StopCondition, Exception]],
     search_strategy: Callable[..., Either[SearchStrategy, Exception]]
)->SearchResults:
    return search_space().bind(
        lambda space: search_strategy().bind(
            lambda strategy: stop_condition().bind(
                lambda until: Right(SearchRequest(space, strategy, until))   
            )
        )
    ).either(lambda e: SearchFailed(error=e), lambda r: do_search(r))