from __future__ import annotations
from typing import List, Optional, Callable, Dict, Union
from collections import OrderedDict
from dataclasses import dataclass
from pymonad.either import Either
import numpy as np
from swarmist.core.dictionary import Selection, UpdateContext, PosEditor, Recombination, Condition, IReference, IReferences, Pos, GroupInfo, Update, Order
from swarmist.core.errors import assert_at_least, assert_not_null, assert_callable
from swarmist.recombination import RecombinationMethods

def select(selection: Callable[...,Selection])->UpdateBuilder:
      def callback()->Selection:
            param = "Selection method"
            assert_not_null(selection, param)
            f = selection()
            assert_callable(f, param)
            return f
      return UpdateBuilder(selection=callback)

def all()->Selection:
      f: Selection = lambda info: info.all()
      return lambda: f

def roulette(size: int = None)->Callable[...,Selection]:
      def callback()->Selection:
            f: Selection = lambda info: info.pick_roulette(size=size if size else info.size())
            assert_at_least(size, 1, "Size of agents to pick with roulette")
            return f
      return callback

def with_probability(p: float = .25)->Callable[..., Selection]:
      def callback()->Selection:
            f: Selection = lambda info: info.filter(lambda _: np.random.uniform() < p)
            return f
      return callback
            
def random(size: int = None)->Callable[..., Selection]:
      def callback()->Selection:
            f: Selection = lambda info: info.pick_random(size)
            assert_at_least(size, 1, "Size of agents to pick randomly")
            return f
      return callback

def filter(condition: Condition, size: int = None)->Callable[..., Selection]:
      def callback()->Selection:
            param = "When condition"
            f: Selection = lambda info: info.filter(size, condition)
            assert_not_null(condition, param)
            assert_at_least(size, 1, param)
            assert_callable(condition, param)
            return f
      return callback

def do_aggregate(
      selection: Callable[..., Selection], 
      key: Union[Order, str])->Callable[..., Selection]:
      select_param = "Selection method"
      aggr_param = "Aggregator method"
      assert_not_null(selection, select_param)
      assert_not_null(key, aggr_param)
      assert_callable(selection, select_param)
      method = key if callable(key) else lambda agent: getattr(agent, key)
      def callback(info: GroupInfo):
            agents = selection(info)
            if len(agents) > 0:
                  return method(agents)
            return []
      return lambda: callback

def max(selection: Callable[..., Selection], key: Union[Order, str] = "fit")->Callable[..., Selection]:
      return do_aggregate(
            selection, 
            lambda agents: max(agents, key=key)
      )

def min(selection: Callable[..., Selection], key: Union[Order, str] = "fit")->Callable[..., Selection]:
      return do_aggregate(
            selection, 
            lambda agents: min(agents, key=key)
      )

PosHelperResult = IReference | IReferences | Pos
PosHelper = Callable[[UpdateContext], PosHelperResult] 
UpdateArgs = Dict[str, Union[PosHelper, Recombination, Condition]]

class UpdateBuilder:

      def __init__(self, selection: Callable[..., Selection]):
            self.selection = selection
            self.recombinator: Recombination = RecombinationMethods().replace_all()
            self.helpers: OrderedDict[str, int] = OrderedDict()
            self.pos_editor: PosEditor = None
            self.where: Optional[Condition] = None

      def update(self, **kwargs: PosHelper)->UpdateBuilder:
            self.pos_editor = kwargs.pop("pos")
            for key, value in kwargs.items():
                  self.helpers[key] = value
            return self
      
      def recombinant(self, recombinator: Recombination)->UpdateBuilder:
            self.recombinator = recombinator
            return self
      
      def where(self, condition: Condition)->UpdateBuilder:
            self.condition = condition
      
      def get(self)->Update:
            def update(ctx: UpdateContext):
                  for key, value in self.helpers.items():
                        ctx.set(key, value(ctx))
                  return self.pos_editor(ctx)
            return Update(
                  method=update,
                  selection=self.selection(),
                  where=self.where
            )