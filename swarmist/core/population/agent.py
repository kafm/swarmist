from __future__ import annotations
from typing import Optional
from dataclasses import replace
import numpy as np
from ..dictionary import *
from ..executor import SearchExecutor

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

def do_apply(new_agent: Agent, old_agent: Agent, executor: SearchExecutor)->Agent: 
    #TODO check if other algorithms works like this
    pos = executor.clip(new_agent.pos)
    delta = pos - old_agent.pos 
    fit = executor.evaluate(pos)
    improved = fit < new_agent.fit
    best = pos 
    trials = 0
    if not improved:
        best = new_agent.best
        fit = new_agent.fit
        trials = new_agent.trials + 1 
    return replace(
        new_agent, 
        delta=delta,
        pos=pos,
        fit=fit,
        best=best,
        trials=trials,
        improved=improved
    )

def update_agent(update: Update, agent: Agent, info: GroupInfo, executor: SearchExecutor)->Agent:    
    where = update.where 
    new_agent = update.method(get_update_context(agent, info))
    candidate = do_apply(
        new_agent=new_agent, 
        old_agent=agent, 
        executor=executor
    )
    return candidate if not where or where(candidate) else agent

def get_update_context(agent: Agent, info: GroupInfo, replace: bool = False)->UpdateContext:
    return UpdateContextWrapper(agent, info, replace).getContext()

class UpdateContextWrapper:
    def __init__(self, agent: Agent, info: GroupInfo, replace:bool=False):
        self.agent = agent
        self.info = info
        self.replace = replace
        self.picked: AgentList = []

    def getContext(self)->UpdateContext:
        return UpdateContext(
            agent=self.agent,
            all = self.info.all,
            size = self.info.size,
            fits = self.info.fits,
            probs = self.info.probs,
            filter = self.info.filter, 
            map = self.info.map,
            best = self._best,
            worse = self._worse,
            pick_random = self._pick_random,
            pick_roulette = self._pick_roulette
        )
    
    def _best(self, k: Optional[int] = None)->Union[Agent, AgentList]:
        return self.info.best(k)
    
    def _worse(self, k: Optional[int] = None)->Union[Agent, AgentList]:
        return self.info.worse(k)
    
    def _pick_random(self, k: Optional[int] = None)-> OneOrMoreAgents:
        return self.info.pick_random(self.info.all(),size=k) #TODO deal with exclude

    def _pick_roulette(self, k: Optional[int] = None)->OneOrMoreAgents:
        return self.info.pick_random(self.info.all(),size=k, p=self.info.probs) #TODO deal with exclude