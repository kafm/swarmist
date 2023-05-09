from __future__ import annotations
from typing import List, Optional, Callable
from pymonad.either import Either
from .dictionary import *
from .errors import try_catch, assert_not_null, assert_not_empty, assert_at_least, assert_callable, assert_number

def size(val: int)->Callable[..., int]:
      def callback()->int:
            param = "Population size"
            assert_not_null(val, param)
            assert_at_least(val, 2, param)
            return val
      return callback

def init(f: PosGenerationMethod)->Callable[..., PosGenerationMethod]:
      def callback()->PosGenerationMethod:
            param = "Position initialization method"
            assert_not_null(f, param)
            assert_callable(f, param)
            return f
      return callback

def topology(f: TopologyBuilder)->Callable[..., TopologyBuilder]:
      def callback()->TopologyBuilder:
            assert_callable(f, "Topology builder method")
            return f
      return callback
    
def all()->Callable[...,SelectionMethod]:
      f: SelectionMethod = lambda info: info.all()
      return lambda: f

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
            assert_callable(condition, param)
            return f
      return callback
      
def select(method: Callable[..., SelectionMethod])->Callable[..., SelectionMethod]:
      def callback()->SelectionMethod:
            param = "Selection method"
            assert_not_null(method, param)
            f = method()
            assert_callable(f, param)
            return f
      return callback

def apply(method: UpdateMethod)->Callable[..., UpdateMethod]:
      def callback()->UpdateMethod:
            param = "Update method"
            assert_not_null(method, param)
            assert_callable(method, param)
            return method
      return callback

def where(*conditions: Condition)->Callable[..., Optional[Condition]]:
      def compose_conditions(conditions: List[Condition], f: Optional[Condition] = None)->Condition:
            if len(conditions) > 0:
                  c = conditions.pop()
                  if f: 
                      c = lambda a: f(a) and c(a)
                  return compose_conditions(conditions, c)
      if not conditions:
            return lambda: None
      return lambda: compose_conditions([c for c in conditions 
                        if c and assert_callable(c, "Condition method")])

def update(
      selection: Callable[..., SelectionMethod],
      method: Callable[..., UpdateMethod],
      where: Optional[Callable[..., Condition]] = None
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
      init: Callable[..., PosGenerationMethod],
      topology: Optional[Callable[..., Topology]],
      *update_pipeline: Callable[..., Update]
)->Either[Callable[..., SearchStrategy], Exception]:
      return lambda: try_catch(
            lambda: SearchStrategy(
                  initialization=Initialization(
                        population_size=size(),
                        generate_pos=init(),
                        topology=None if not topology else topology()
                  ),
                  update_pipeline=get_update_pipeline(*update_pipeline)
            )
      )