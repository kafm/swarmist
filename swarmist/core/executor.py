from __future__ import annotations
from typing import Optional
from pymonad.either import Right, Either
from dataclasses import replace
import sys
import numpy as np
from .dictionary import *
from .errors import SearchEnded, try_catch


class SearchExecutor:
    def __init__(self, 
        fit_func: FitnessFunctionDef,
        ndims: int, 
        bounds: Bounds,
        replace: bool,
        constraints: Optional[ConstraintsChecker],
        max_gen: Optional[int],
        min_fit: Optional[float], 
        max_evals:  Optional[int],
    ):
        self.ctx = ExecutionContext()
        self.search_ctx = SearchContext(
             self.evaluate, 
             self.clip,
             ndims, 
             bounds,
             replace=replace
        )
        self.fit_func:FitnessFunction = fit_func
        self.constraints = constraints
        self.min_fit = min_fit
        self.max_evals = max_evals
        self.max_gen = max_gen
        self.generations:List[ExecutionContext] = []
        self.error = None

    def context(self)->SearchContext:
        return self.search_ctx
    
    def clip(self, pos: Pos)->Pos:
        bounds = self.search_ctx.bounds
        cs = self.constraints
        new_pos = np.clip(pos, bounds.min, bounds.max)
        return new_pos if not cs else cs(new_pos)

    def evaluate(self, pos :Pos)->Fit:
        self._assert_max_evals()
        self._assert_min_fit()
        fit = self.fit_func(pos)
        curr_eval = self.ctx.curr_eval + 1
        best = self.ctx.best
        self.ctx = replace(
            self.ctx, 
            best=Evaluation(pos, fit) if fit < best.fit else best, 
            curr_eval=curr_eval,
            arquived=False
        )    
        return fit 
    
    #TODO find better design
    def assert_best(self, pos: Pos, fit: Fit):
        if fit < self.ctx.best.fit:
            self.ctx = replace(
                self.ctx,
                best=Evaluation(pos, fit),
                arquived=False
            )    

    def next(self)->Either[ExecutionContext, Exception]:
        res = try_catch(self._assert_max_gen)
        if res.is_right():
            self._arquive_execution()
            self.ctx = replace(self.ctx,
                #TODO find better way #best=Evaluation(se),
                curr_gen=self.ctx.curr_gen+1
            )
            return Right(self.ctx)
        else:
            self.error = res.either(lambda e: e, None)
        return res

    def results(self)->List[ExecutionContext]:
        if not self.ctx.arquived:
            self._arquive_execution()
        return self.generations
    
    def _arquive_execution(self):
        self.ctx = replace(self.ctx, arquived=True)
        self.generations.append(self.ctx)
    
    def _assert_max_evals(self):
        if self.max_evals and self.ctx.curr_eval >= self.max_evals:
            raise SearchEnded("Maximum number of evaluations reached")
        
    def _assert_min_fit(self):
        if self.min_fit and self.ctx.best.fit <= self.min_fit:
            raise SearchEnded("Fitness goal reached")
        
    def _assert_max_gen(self):
        if self.max_gen and self.ctx.curr_gen >= self.max_gen:
            raise SearchEnded("Maximum number of generations reached")