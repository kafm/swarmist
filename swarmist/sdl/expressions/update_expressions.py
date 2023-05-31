from lark import v_args
from collections import OrderedDict
from swarmist.recombination import RecombinationMethods
from swarmist import (
    all,
    filter,
    order,
    limit,
    roulette,
    random,
    with_probability,
    select,
)
from .expressions import Expressions


@v_args(inline=True)
class UpdateExpressions(Expressions):
    def all_selection(self, size=None, order_by=None):
        return select(self._append_order_by_and_size(all(), order_by, size))

    def filter_selection(self, size=None, where=None, order_by=None):
        return select(self._append_order_by_and_size(filter(where), order_by, size))

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
        return (key, value)
