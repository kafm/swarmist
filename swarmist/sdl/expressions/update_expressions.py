from lark import v_args
from typing import Optional, cast, List
from dataclasses import dataclass
from collections import OrderedDict
from swarmist.core.dictionary import Condition, Update
from swarmist.recombination import RecombinationMethods, RecombinationMethod 
from swarmist import (
    all,
    filter,
    order,
    limit,
    roulette,
    random,
    with_probability,
    select,
    UpdateBuilder,
)
from .expressions import Expressions

@dataclass(frozen=True)
class UpdateTail:
    recombination: RecombinationMethod
    update_pos: OrderedDict
    when: Optional[Condition]

@v_args(inline=True)
class UpdateExpressions(Expressions):

    def __init__(self):
        self._pipeline: List[Update] = []

    def get_pipeline(self):
        return self._pipeline

    def update(self, selection: UpdateBuilder, update_tail: UpdateTail):
        self._pipeline.append(
            select(selection)
                .update(**update_tail.update_pos)
                .recombinant(update_tail.recombination)
                .where(update_tail.when)
                .get()
        )

    def replace_all_pos(self, update_pos, when=None):
        return UpdateTail(
            RecombinationMethods().replace_all(),
            update_pos,
            when=when
        )
        
    def recombine_pos(self, recombination, update_pos, when=None):
        return UpdateTail(
            recombination=recombination,
            update_pos=update_pos,
            when=when
        )

    def all_selection(self, size=None, order_by=None):
        return self._append_order_by_and_size(all(), order_by, size)

    def filter_selection(self, size=None, where=None, order_by=None):
        return self._append_order_by_and_size(filter(where), order_by, size)

    def _append_order_by_and_size(self, selection, order_by, size):
        if order_by:
            selection = order_by(selection)
        if size:
            selection = limit(selection)
        return selection

    def roulette_selection(self, size=None):
        return roulette(size=size)

    def random_selection(self, size=None):
        return random(size=size)

    def probabilistic_selection(self, probability, size=None):
        return with_probability(p=probability, size=size)

    def selection_size(self, size=None):
        return size

    def order_by(self, field, asc_desc=False):
        return lambda selection: order(selection, field, reverse=asc_desc)

    def reverse_order(self):
        return True

    def binomial_recombination(self, p: float):
        return RecombinationMethods().binomial(p)

    def exponential_recombination(self, p: float):
        return RecombinationMethods().exponential(p)

    def with_probability_recombination(self, p: float):
        return RecombinationMethods().k_with_probability(p)

    def random_recombination(self, size: int):
        return RecombinationMethods().k_random(size)

    def init_random_recombination(self):
        return RecombinationMethods().get_new()

    def update_pos(self, *args):
        return OrderedDict({arg[0]: arg[1] for arg in args})

    def set_update_var(self, key, value):
        if cast(str, key).lower() == "pos":
            return ("pos", value)
        return (key, value)