from __future__ import annotations
from typing import List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from oslash import Left, Right, Either
import numpy as np
import inspect
from Dictionary import *
from Helpers import try_catch, assert_not_null, assert_not_empty, assert_at_least, assert_function_signature
from functools import partial

#TODO after we need to check if a centroid, reference, perturbation, etc, formula is not better

def size(val: int)->Callable[..., int]:
      def callback():
            param = "Population size"
            assert_not_null(val, param)
            assert_at_least(val, 2, param)
            return val
      return callback

def init(f: InitializationMethod)->Callable[..., InitializationMethod]:
      def callback():
            param = "Initialization method"
            assert_not_null(f, param)
            assert_function_signature(f, InitializationMethod, param)
            return f
      return callback

def topology(f: TopologyBuilder)->Callable[..., TopologyBuilder]:
      def callback():
            param = "Topology builder method"
            assert_function_signature(f, TopologyBuilder, param)
            return f
      return callback
    
def all()->Callable[...,SelectionMethod]:
      f: SelectionMethod = lambda info: info.all()
      return lambda: f()

def roulette(size: int = None)->Callable[...,SelectionMethod]:
      def callback()->SelectionMethod:
            f: SelectionMethod = lambda info: info.pick_roulette(size)
            assert_at_least(size, 1, "Size of agents to pick with roulette")
            return f
      return callback
            
def random(size: int = None)->Callable[...,SelectionMethod]:
      def callback()->SelectionMethod:
            f: SelectionMethod = lambda info: info.pick_random(size)
            assert_at_least(size, 1, "Size of agents to pick randomly")
            return f
      return callback

def when(condition: Condition, size: int = None)->Callable[..., SelectionMethod]:
      def callback()->SelectionMethod:
            param = "When condition"
            f: SelectionMethod = lambda info: info.filter(size, condition)
            assert_not_null(condition, param)
            assert_at_least(size, 1, param)
            assert_function_signature(condition, Condition, param)
            return f
      return callback
      
def select(method: Callable[..., SelectionMethod])->Callable[..., SelectionMethod]:
      def callback()->SelectionMethod:
            param = "Selection method"
            assert_not_null(method, param)
            f = method()
            assert_function_signature(f, Callable[..., SelectionMethod], param)
            return f
      return callback

def apply(method: UpdateMethod)->Callable[..., UpdateMethod]:
      def callback()->UpdateMethod:
            param = "Update method"
            assert_not_null(method, param)
            assert_function_signature(method, UpdateMethod, param)
            return method
      return callback

def where(*conditions: Condition)->Callable[..., Optional[Conditions]]:
      return lambda: [c for c in conditions 
                        if c and assert_function_signature(c, Condition, "Condition method")]

def update(
      method: Callable[..., UpdateMethod],
      selection: Callable[..., SelectionMethod],
      where: Optional[Callable[..., Conditions]] = None
)->Callable[..., Update]:
      return lambda: Update(
                  method(),
                  selection(),
                  where and where()
            )

def get_update_pipeline(*update_pipeline: Callable[..., Update])->List[Update]:
      assert_not_empty(update_pipeline, "Update pipeline")
      return [u() for u in update_pipeline]

def using(
      size: Callable[..., int],
      init: Callable[..., InitializationMethod],
      topology: Callable[..., Topology],
      *update_pipeline: Callable[..., Update]
)->Either[Callable[..., SearchStrategy], Exception]:
      return lambda: try_catch(
            lambda: SearchStrategy(
                  init=Init(
                        population_size=size(),
                        method=init(),
                        topology=None if not topology else topology()
                  ),
                  update_pipeline=get_update_pipeline(update_pipeline)
            )
      )