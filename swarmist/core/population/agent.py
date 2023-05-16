from __future__ import annotations
from typing import Optional
from dataclasses import replace
import numpy as np
from ..dictionary import *
from ..executor import SearchExecutor

def get_fits(agents: AgentList)->List[float]:
    return [a.fit for a in agents]

def fit_to_prob(fits: List[float])->List[int]:
    min_fit = min(fits)
    max_fit = max(fits)
    total = 0
    n = len(fits)
    nfits = np.zeros(n)
    if min_fit == max_fit: 
        return np.full(n, 1/n)
        #TODO investigate: print(f"fits={fits}")
    for i in range(n):
        fit = (max_fit - fits[i]) / (max_fit - min_fit)
        nfits[i] = fit
        total += fit
    return np.divide(nfits, total)

def create_agent(pos_generator: PosGenerationMethod, index: int, ctx: SearchContext)->Agent:
    pos = ctx.clip(pos_generator(ctx))
    fit = ctx.evaluate(pos)  
    return Agent(
        index=index,
        ndims=ctx.ndims,
        pos=pos,
        best=pos,
        delta=np.zeros(ctx.ndims),
        fit=fit,
        trials=0,
        improved=True
    )

def do_apply(agent: Agent, old_agent: Agent ,executor: SearchExecutor)->Agent: 
    #TODO check if other algorithms works like this
    pos = executor.clip(agent.pos)
    delta = pos - old_agent.pos 
    fit = executor.evaluate(pos)
    improved = fit < agent.fit
    best = pos 
    trials = 0
    if not improved:
        best = old_agent.best
        fit = old_agent.fit
        trials = old_agent.trials + 1 
        #print(f"agent {new_agent.index} did not improve")
    return replace(
        agent, 
        delta=delta,
        pos=pos,
        fit=fit,
        best=best,
        trials=trials,
        improved=improved
    )

def get_update_context(agent: Agent, info: GroupInfo, bounds: Bounds)->UpdateContext:
    return UpdateContextWrapper(agent, info, bounds).getContext()

def update_agent(update: Update, agent: Agent, info: GroupInfo, executor: SearchExecutor)->Agent:    
    where = update.where 
    new_agent = update.method(get_update_context(agent, info, executor.context().bounds))
    candidate = do_apply(
        agent=new_agent, 
        old_agent=agent,
        executor=executor
    )
    return candidate if not where or where(candidate) else replace(agent, trials=candidate.trials)

class UpdateContextWrapper:
    def __init__(self, agent: Agent, info: GroupInfo, bounds: Bounds):
        self.agent = agent
        self.info = info
        self.bounds = bounds
        self.picked: List[int] = [agent.index]

    def getContext(self)->UpdateContext:
        return UpdateContext(
            agent=self.agent,
            bounds = self.bounds,
            all = self.info.all,
            size = self.info.size,
            fits = self.info.fits,
            probs = self.info.probs,
            filter = self.info.filter, 
            map = self.info.map,
            best = self._best,
            worse = self._worse,
            pick_random = self._pick_random,
            pick_roulette = self._pick_roulette,
            pick_random_unique = self._pick_random_unique,
            pick_roulette_unique = self._pick_roulette_unique
        )
    
    def _best(self, k: Optional[int] = None)->OneOrMoreAgents:
        agents = self.info.best(k)
        self._append_picked(agents)
        return agents
    
    def _worse(self, k: Optional[int] = None)->OneOrMoreAgents:
        agents =self.info.worse(k)
        self._append_picked(agents)
        return agents
        
    def _pick_random_unique(self, k: Optional[int] = None, replace: Optional[bool] = False)-> OneOrMoreAgents:
        agents = list(self.info.filter(lambda a: a.index not in self.picked))
        agents = np.random.choice(agents, size=k, replace=replace)
        self._append_picked(agents)
        return agents
    
    def _pick_random(self, k: Optional[int] = None, replace: Optional[bool] = False)-> OneOrMoreAgents:
        agents = self.info.pick_random(replace=replace,size=k)
        self._append_picked(agents)
        return agents
    
    def _pick_roulette_unique(self, k: Optional[int] = None, replace: Optional[bool] = True)-> OneOrMoreAgents:
        agents = self.info.filter(lambda a: a.index not in self.picked)
        agents = np.random.choice(
            agents, size=k, replace=replace, p=fit_to_prob(get_fits(agents))
        )
        self._append_picked(agents)

    def _pick_roulette(self, k: Optional[int] = None, replace: Optional[bool] = True)->OneOrMoreAgents:
        agents = self.info.pick_random(size=k, replace=replace, p=self.info.probs)
        self._append_picked(agents)
        return agents
    
    def _append_picked(self, agents: OneOrMoreAgents):
        _agents = [agents] if isinstance(agents, Agent) else agents
        for agent in _agents: 
            self.picked.append(agent.index)