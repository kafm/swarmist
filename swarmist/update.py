from __future__ import annotations
from typing import List, Optional, Callable
from dataclasses import dataclass
from pymonad.either import Either
import numpy as np

@dataclass(frozen=True)
class Update:
    method: Callable[[UpdateContext], Agent]
    selection: SelectionMethod
    where: Condition = None





def all()->Callable[...,SelectionMethod]:
      f: SelectionMethod = lambda info: info.all()
      return lambda: f

def roulette(size: int = None)->Callable[...,SelectionMethod]:
      def callback()->SelectionMethod:
            f: SelectionMethod = lambda info: info.pick_roulette(size=size if size else info.size())
            assert_at_least(size, 1, "Size of agents to pick with roulette")
            return f
      return callback

def with_probability(p: float = .25)->Callable[..., SelectionMethod]:
      def callback()->SelectionMethod:
            f: SelectionMethod = lambda info: info.filter(lambda _: np.random.uniform() < p)
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

def aggregator(
      selection: Callable[..., SelectionMethod], 
      method: Callable[[AgentList],Agent])->Callable[..., SelectionMethod]:
      select_param = "Selection method"
      aggr_param = "Aggregator method"
      assert_not_null(selection, select_param)
      assert_not_null(method, aggr_param)
      assert_callable(selection, select_param)
      assert_callable(method, aggr_param)
      def callback(info: GroupInfo):
            agents = selection(info)
            if len(agents) > 0:
                  return [method(agents)]
            return []
      return lambda: callback

def biggest(key: OrderingMethod, selection: Callable[..., SelectionMethod] = all)->Callable[..., SelectionMethod]:
      return aggregator(
            selection, 
            lambda agents: max(agents, key=key)
      )

def smallest(key: OrderingMethod, selection: Callable[..., SelectionMethod] = all)->Callable[..., SelectionMethod]:
      return aggregator(
            selection, 
            lambda agents: min(agents, key=key)
      )
      
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
      topology: Optional[Callable[..., TopologyBuilder]],
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