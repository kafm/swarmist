from __future__ import annotations
from typing import List, Optional, Callable, Dict, Union
from functools import partial
from collections import OrderedDict
from dataclasses import replace
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

def all()->Callable[...,Selection]:
      f: Selection = lambda info: info.all()
      return lambda: f

def order(selection: Selection, key: Order, reverse: bool = False)->Callable[...,Selection]:
      def callback(info: GroupInfo)->Selection:
            method = key if callable(key) else lambda agent: getattr(agent, key)
            return selection(info).sort(key=method, reverse=reverse)
      return lambda: callback

def limit(selection: Selection, size: int)->Callable[...,Selection]:
      def callback(info: GroupInfo)->Selection:
            els = selection(info)
            if not els:
                  return []
            elif len(els) <= size:
                  return els
            return els[0:size]
      return lambda: callback

def roulette(size: int = None)->Callable[...,Selection]:
      def callback()->Selection:
            f: Selection = lambda info: info.pick_roulette(size=min(size,info.size()) if size else info.size())
            return f
      return callback

def with_probability(p: float = .25, size: int = None)->Callable[..., Selection]:
      def callback(info: GroupInfo)->Selection:
            agents = info.filter(lambda _: np.random.uniform() < p)
            if not size or size >= len(agents):
                  return agents
            else:
                  return agents[0:size]
      return lambda: callback
            
def random(size: int = None)->Callable[..., Selection]:
      def callback(info: GroupInfo)->Selection:
            return info.pick_random(size=min(size,info.size()) if size else info.size())
      return lambda: callback

def filter(condition: Condition)->Callable[..., Selection]:
      def callback()->Selection:
            param = "When condition"
            f: Selection = lambda info: info.filter(condition)
            assert_not_null(condition, param)
            assert_callable(condition, param)
            return f
      return callback

# def do_aggregate(
#       selection: Callable[..., Selection], 
#       key: Union[Order, str])->Callable[..., Selection]:
#       select_param = "Selection method"
#       aggr_param = "Aggregator method"
#       assert_not_null(selection, select_param)
#       assert_not_null(key, aggr_param)
#       assert_callable(selection, select_param)
#       method = key if callable(key) else lambda agent: getattr(agent, key)
#       def callback(info: GroupInfo):
#             agents = selection(info)
#             if len(agents) > 0:
#                   return method(agents)
#             return []
#       return lambda: callback

# def max(selection: Callable[..., Selection], key: Union[Order, str] = "fit")->Callable[..., Selection]:
#       return do_aggregate(
#             selection, 
#             lambda agents: max(agents, key=key)
#       )

# def min(selection: Callable[..., Selection], key: Union[Order, str] = "fit")->Callable[..., Selection]:
#       return do_aggregate(
#             selection, 
#             lambda agents: min(agents, key=key)
#       )

PosHelperResult = IReference | IReferences | Pos
PosHelper = Callable[[UpdateContext], PosHelperResult] 
UpdateArgs = Dict[str, Union[PosHelper, Recombination, Condition]]

class UpdateBuilder:

      def __init__(self, selection: Callable[..., Selection]):
            self.selection = selection
            self.recombinator: Recombination = RecombinationMethods().replace_all()
            self.helpers: OrderedDict[str, PosHelper] = OrderedDict()
            self.pos_editor: PosEditor = None
            self.condition: Optional[Condition] = None

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
            return self
      
      def get(self)->Update:
            def update(ctx: UpdateContext):
                  vars: Dict[str, Union[Pos, IReference, IReferences]] = {}
                  _ctx = replace(ctx, vars=vars)
                  for key, value in self.helpers.items():
                        vars[key] = value(_ctx)
                  return self.pos_editor(_ctx)
            return Update(
                  selection=self.selection(),
                  editor=update,
                  recombination=self.recombinator,
                  condition=self.condition
            )